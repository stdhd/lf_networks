from networks.hamiltonian_u_net import HamiltonianUNet
from networks.attention.cross_attention import PreNorm, Attention, FeedForward
from networks.attention.attention_modules import MultiHeadCrossAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

class CAUnet(HamiltonianUNet):
    def __init__(self, in_channels, ca_depth=1, **kwargs):
        super().__init__(in_channels, **kwargs)
        context_dim = 64
        self.up1 = CAup(512, 256, context_dim, ca_depth=ca_depth)
        self.up2 = CAup(256, 128, context_dim, ca_depth=ca_depth)
        self.up3 = CAup(128, 64, context_dim, ca_depth=ca_depth)

    def apply_cross_attention(self, x, context, i):
        return self.ca_mods[i](Y=x, S=context)

    def forward(self, recurrent_state, q, context=None):
        x_orig = recurrent_state
        x0 = self.inc(x_orig)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2, context)
        x = self.up2(x, x1, context)
        x = self.up3(x, x0, context)
        x = self.outc(x)

        if self.final_layer:
            x = self.final_conv(x)
            x = self.final_relu(x)
        return x


class CAup(nn.Module):
    def __init__(self, FromChannels, ToChannels, ContextChannels, ca_depth=1, cross_dim_head=32, latent_dim_head=32, latent_heads=6, cross_heads=6):
        super(CAup, self).__init__()
        self.FromChannels = FromChannels
        self.ToChannels = ToChannels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(FromChannels,
                      ToChannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.GroupNorm(num_channels=ToChannels, num_groups=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(ToChannels,
                      ToChannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True), nn.GroupNorm(num_channels=ToChannels, num_groups=16),
            nn.ReLU(inplace=True))

        attn_dropout = 0
        ff_dropout = 0

        input_dim = ContextChannels
        latent_dim = FromChannels

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads,
                                                   dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        self.attn_layers = nn.ModuleList([])
        for _ in range(ca_depth):
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

    def forward(self, lowerBlock, equalLevel, context):

        lowerBlock = self.up(lowerBlock)

        x1_w = lowerBlock.size(-1)
        context = rearrange(context, 'b c w h -> b (w h) c', b=context.size(0))
        x1 = rearrange(lowerBlock, 'b c w h -> b (w h) c', b=lowerBlock.size(0))

        for cross_attn, cross_ff, self_attns in self.attn_layers:
            x1 = cross_attn(x1, context=context, mask=None) + x1
            x1 = cross_ff(x1) + x1

            for self_attn, self_ff in self_attns:
                x1 = self_attn(x1) + x1
                x1 = self_ff(x1) + x1
        x1 = rearrange(x1, 'b (w h) c -> b c w h', c=self.FromChannels, w=x1_w, b=lowerBlock.size(0))
        x1 = self.conv(x1)

        diffY = equalLevel.size()[2] - x1.size()[2]
        diffX = equalLevel.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([equalLevel, x1], dim=1)
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
