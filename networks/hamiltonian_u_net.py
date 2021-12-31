""" Full assembly of the parts to form the complete network """
""" https://github.com/milesial/Pytorch-UNet/tree/master/unet """
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.attention.cross_attention import PreNorm, Attention, FeedForward
from networks.attention.attention_modules import MultiHeadSelfAttention
from networks.attention.attention_modules import MultiHeadCrossAttention
from einops import rearrange, repeat

class HamiltonianUNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.final_layer = kwargs.get('final_conv', False)
        self.bilinear = kwargs.get('bilinear', False)
        self.use_attention = kwargs.get('use_attention', False)
        self.use_self_attention = kwargs.get('use_self_attention', False)
        self.final_fc = kwargs.get('final_fc', False)
        self.intermediate_fc = kwargs.get('intermediate_fc', False)
        self.use_cross_attention = kwargs.get('use_cross_attention', False)
        latent_dim = 32
        input_dim = 64*64
        cross_heads = 12
        cross_dim_head = 12
        attn_dropout = 0.
        ff_dropout = 0
        latent_heads = 12
        latent_dim_head = 32

        if self.use_self_attention:
            self.sa_mods = nn.ModuleList([Attention(64, heads=8, dim_head=128), Attention(128, heads=32, dim_head=256), Attention(256, heads=64, dim_head=512)])

        self.n_channels = in_channels
        acivations = {'relu': torch.nn.ReLU(inplace=True), 'leakyrelu': torch.nn.LeakyReLU(inplace=True),
                      'elu': torch.nn.ELU(inplace=True), 'tanh': torch.nn.Tanh()}
        self.activation = acivations[kwargs.get('activation', 'relu')]

        self.inc = DoubleConv(in_channels, 64)
        self.outc = OutConv(64, 64)
        self.down0 = Down(64, 128, activation=self.activation)
        self.down1 = Down(128, 256, activation=self.activation)
        factor = 2 if self.bilinear else 1
        self.down2 = Down(256, 512 // factor, activation=self.activation)

        if 1==0:
            self.up1 = TransformerUp(512, 256)
            self.up2 = TransformerUp(256, 128)
            self.up3 = TransformerUp(128, 64)
        else:
            self.up1 = Up(512, 256 // factor, self.bilinear, activation=self.activation,
                          use_attention=self.use_attention)
            self.up2 = Up(256, 128 // factor, self.bilinear, activation=self.activation,
                          use_attention=self.use_attention)
            self.up3 = Up(128, 64, self.bilinear, activation=self.activation, use_attention=self.use_attention)


        if 1==0:

            get_cross_attn = lambda: PreNorm(latent_dim,
                                             Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                       dropout=attn_dropout), context_dim=input_dim)
            get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
            get_latent_attn = lambda: PreNorm(latent_dim,
                                              Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                        dropout=attn_dropout))
            get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

            self.attn_layers = nn.ModuleList([])
            for _ in range(5):
                self_attns = nn.ModuleList([])
                self_attns.append(nn.ModuleList([
                    get_latent_attn(),
                    get_latent_ff()
                ]))

                self.attn_layers.append(nn.ModuleList([
                    get_cross_attn(),
                    get_cross_ff(),
                    self_attns
                ]))

        if self.final_layer:
            self.final_conv = torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
            self.final_relu = self.activation

    def apply_attention(self, x, i):
        if self.use_self_attention:
            w, h = x.size(-1), x.size(-1)
            x = rearrange(x, 'b c w h -> b (w h) c', w=w, h=h)
            x = self.sa_mods[i](x)
            return rearrange(x, 'b (w h) c -> b c w h', w=w, h=h)
        else:
            return x

    def forward(self, p, q, context=None):
        x_orig = torch.cat((p, q), dim=1) # shape 18, 4
        x0 = self.inc(x_orig)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x2_att = self.apply_attention(x2, 2)
        x = self.up1(x3, x2_att)
        x1_att = self.apply_attention(x1, 1)
        x = self.up2(x, x1_att)
        x0_att = self.apply_attention(x0, 0)
        x = self.up3(x, x0_att)
        x = self.outc(x)

        if 1==0:
            x = self.attn_layers(x)
            for cross_attn, cross_ff, self_attns in self.layers:
                x = cross_attn(x, context=context, mask=None) + x
                x = cross_ff(x) + x

                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x

        if self.final_layer:
            x = self.final_conv(x)
            x = self.final_relu(x)
            return None, x
        else:
            q, p = self.to_phase_space(x)
            return q, p

    @staticmethod
    def to_phase_space(encoding):
        assert encoding.shape[1] % 2 == 0, \
            'The number of in_channels is odd. Cannot split properly.'
        half_len = int(encoding.shape[1] / 2)
        q = encoding[:, :half_len]
        p = encoding[:, half_len:]
        return q, p

""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=torch.nn.ReLU(inplace=True)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.GroupNorm(num_channels=mid_channels, num_groups=16),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_channels=out_channels, num_groups=16),
            activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation=torch.nn.ReLU):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activation=torch.nn.ReLU, use_attention=False, use_cross_attention=False):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.x1att_conv = torch.nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(1, 1))
            self.x2att_conv = torch.nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=(1, 1))

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # BILINEAR:
        #torch.Size([2, 512, 1, 1])
        #torch.Size([2, 512, 2, 2])

        # NICHT BILINEAR:
        #torch.Size([2, 512, 1, 1])
        #torch.Size([2, 256, 2, 2])

        #exit()

        if self.use_attention:
            x1_att = self.x1att_conv(x1)
            x2_att = self.x2att_conv(x2)
            sum = x1_att + x2_att
            sum = self.relu(sum)
            attention = self.sigmoid(sum)
            x2 = attention * x2

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x





