import torch.nn as nn, torch
import torchvision.models as models


class VGGStartFrameEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, **kwargs
                 ):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True).features
        self.fc1 = torch.nn.Linear(2048, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.model(x).view(-1, 512*2*2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.reshape((-1, 32, 8, 8))
        return x


class DummyStartFrameEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        x = torch.randn((x.size(0), 32, 8, 8), device=x.device)
        return x


class StartFrameEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 act_func=nn.ReLU(),
                 dtype=torch.float,
                 **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, affine=False, track_running_stats=True)
        self.activation = torch.nn.LeakyReLU()
        paddings = [int(k / 2) for k in kernel_sizes]

        self.start_resnet = nn.Conv2d(in_channels,
                                      n_filters[0],
                                      kernel_size=kernel_sizes[0],
                                      stride=strides[0],
                                      padding=paddings[0]
                                      )

        self.hidden_layers = nn.ModuleList([
            nn.Conv2d(in_channels=n_filters[i],
                      out_channels=n_filters[i + 1],
                      kernel_size=kernel_sizes[i + 1],
                      padding=paddings[i + 1],
                      stride=strides[i + 1]) for i in range(hidden_conv_layers)

        ])

        self.norms = nn.ModuleList([
            nn.BatchNorm2d(n_filters[i + 1]) for i in range(hidden_conv_layers)

        ])

        self.out = nn.Conv2d(in_channels=n_filters[-1],
                                  out_channels=out_channels,
                                  kernel_size=kernel_sizes[-1],
                                  padding=paddings[-1],
                                  stride=strides[-1])

    def forward(self, x):
        x = self.norm(x)
        #x = self.activation(x) # ADDED 1
        x = self.start_resnet(x)
        #x = self.activation(x)  # ADDED 1
        for i, l in enumerate(self.hidden_layers):
            x = self.hidden_layers[i](x)
            #x = self.activation(x
            #x = self.norms[i](x)

        x = self.out(x)
        return x


StartEncoders = {
    'VGG': VGGStartFrameEncoder,
    'standard': StartFrameEncoder,
    'dummy': DummyStartFrameEncoder,

}


def get_encoder(**params):
    return StartEncoders[params.get('type', 'standard')](**params)
