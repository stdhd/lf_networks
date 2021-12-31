"""This module contains the implementation of a decoder network, that applies 3 residual blocks
to the input abstract position q. In the paper q is a (16, 4, 4) tensor that can be seen as a 4x4
image with 16 channels, but here other sizes may be used.
"""

import torch
from torch import nn
import torch.nn.functional as F
from networks.attention.attention_modules import MultiHeadSelfAttention

class ResidualBlock(nn.Module):
    """A residual block that up-samples the input image by a factor of 2.
    """
    def __init__(self, in_channels, n_filters=64, kernel_size=3, dtype=torch.float):
        """Instantiate the residual block, composed by a 2x up-sampling and two convolutional
        layers.

        Args:
            in_channels (int): Number of input channels.
            n_filters (int): Number of filters, and thus output channels.
            kernel_size (int): Size of the convolutional kernels.
            dtype (torch.dtype): Type to be used in tensors.
        """
        super().__init__()
        self.channels = in_channels
        self.n_filters = n_filters
        padding = int(kernel_size / 2)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_filters,
            out_channels=n_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        if in_channels != n_filters:
            self.dim_match_conv = nn.Conv2d(in_channels=in_channels,
                                            out_channels=n_filters,
                                            kernel_size=1,
                                            padding=0)
        self.leaky_relu = nn.LeakyReLU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()
        self.type(dtype)

    def forward(self, x):
        """Apply 2x up-sampling, followed by two convolutional layers with leaky relu. A sigmoid
        activation is applied at the end.

        TODO: Should we use batch normalization? It is often common in residual blocks.
        TODO: Here we apply a convolutional layer to the input up-sampled tensor if its number
            of channels does not match the convolutional layer channels. Is this the correct way?

        Args:
            x (torch.Tensor): Input image of shape (N, C, H, W) where N is the batch size and C
                is the number of in_channels.

        Returns:
            A torch.Tensor with the up-sampled images, of shape (N, n_filters, H, W).
        """
        x = self.upsample(x)
        residual = self.dim_match_conv(
            x) if self.channels != self.n_filters else x
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.sigmoid(x + residual)
        return x


class DecoderNet(nn.Module):
    """The Decoder network, that takes a latent encoding of shape (in_channels, H, W)
    and produces the output image by applying 3 ResidualBlock modules and a final 1x1 convolution.
    Each residual block up-scales the image by 2, and the convolution produces the desired number
    of output channels, thus the output shape is (out_channels, H*2^3, W*2^3).
    """

    def __init__(self,
                 in_channels,
                 out_channels=3,
                 n_residual_blocks=None,
                 n_filters=None,
                 kernel_sizes=None,
                 dtype=torch.float,
                 include_spade=True,
                 use_self_attention=False):
        """Create the decoder network composed of the given number of residual blocks.

        Args:
            in_channels (int): Number of input encodings channels.
            out_channels (int): Number output image channels (1 for grayscale, 3 for RGB).
            n_residual_blocks (int): Number of residual blocks in the network.
            n_filters (list): List where the i-th element is the number of filters for
                convolutional layers for the i-th residual block, excluding the output block.
                Therefore, n_filters must be of length n_residual_blocks - 1
            kernel_sizes(list): List where the i-th element is the kernel size of convolutional
                layers for the i-th residual block.
        """
        super().__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = MultiHeadSelfAttention(in_channels)

        filters = [in_channels] + n_filters
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=int(filters[i]),
                n_filters=int(filters[i + 1]),
                kernel_size=int(kernel_sizes[i]),
                dtype=dtype
            ) for i in range(n_residual_blocks)
        ])
        if include_spade:
            self.spade_blocks = nn.ModuleList([
                Spade(
                    num_features=int(filters[i + 1])
                ) for i in range(n_residual_blocks)
            ])
        else:
            self.spade_blocks = nn.ModuleList([])
        self.out_conv = nn.Conv2d(
            in_channels=filters[-1],
            out_channels=out_channels,
            kernel_size=kernel_sizes[-1],
            padding=int(kernel_sizes[-1] / 2)  # To not resize the image
        )
        self.sigmoid = nn.Sigmoid()
        self.type(dtype)

    def forward(self, x, start_frame):

        """Apply the three residual blocks and the final convolutional layer.

        Args:
            x (torch.Tensor): Tensor of shape (N, in_channels, H, W) where N is the batch size.

        Returns:
            Tensor of shape (out_channels, H * 2^3, W * 2^3) with the reconstructed image.
        """
        if self.use_self_attention:
            x = self.attention(x)

        for i, layer in enumerate(self.residual_blocks):
            x = layer(x)
            if len(self.spade_blocks) > i:
                x = self.spade_blocks[i](x, start_frame)
        x = self.sigmoid(self.out_conv(x))
        return x


class Spade(nn.Module):
    def __init__(self, num_features, num_groups=16):
        super().__init__()
        name = 'instance'
        self.num_features = num_features
        while self.num_features % num_groups != 0:
            num_groups -= 1
        if name == 'BN' or name == 'batch':
            self.norm = nn.BatchNorm2d(num_features, affine=False, track_running_stats=True)
        elif name == 'group' or name == 'Group':
            self.norm = nn.GroupNorm(num_groups, num_features, affine=False)
        elif name == 'instance':
            self.norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=True)
        else:
            raise NotImplementedError('Normalization Method not implemented: ', name)

        self.conv       = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv_gamma = nn.Conv2d(128, num_features, 3, 1, 1)
        self.conv_beta  = nn.Conv2d(128, num_features, 3, 1, 1)
        self.activate   = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        normalized = self.norm(x)
        y = F.interpolate(y, mode='bilinear', size=x.shape[-2:], align_corners=True)
        y = self.activate(self.conv(y))
        gamma = self.conv_gamma(y)#.unsqueeze(2).repeat_interleave(x.size(2), 2)
        beta  = self.conv_beta(y)#.unsqueeze(2).repeat_interleave(x.size(2), 2)
        return normalized * (1 + gamma) + beta
