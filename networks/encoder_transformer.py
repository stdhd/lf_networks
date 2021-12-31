"""This module contains the implementation of the Encoder step of the Hamiltonian Generative Networks
paper. The encoder maps the input sequence of frames into a latent distribution and samples a Tensor z
from it using the re-parametrization trick.
"""

import torch
import torch.nn as nn
from networks.self_attention_cv import ViT

class EncoderTransformerNet(nn.Module):
    """Implementation of the encoder network, that encodes the input frames sequence into a
    distribution over the latent space and samples with the common reparametrization trick.

    The network expects the images to be concatenated along channel dimension. This means that if
    a batch of sequences has shape (batch_size, seq_len, channels, height, width) the network
    will accept an input of shape (batch_size, seq_len * channels, height, width).
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 6,
        'n_filters': [32, 64, 64, 64, 64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3, 3, 3, 3, 3],
        'strides': [1, 1, 1, 1, 1, 1, 1, 1],
    }

    def __init__(self,
                 seq_len,
                 in_channels,
                 out_channels,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 act_func=nn.ReLU(),
                 type=None,
                 dtype=torch.float):
        """Instantiate the convolutional layers that compose the input network with the
        appropriate shapes.

        If K is the total number of layers, then hidden_conv_layers = K - 2. The length of
        n_filters must be K - 1, and that of kernel_sizes and strides must be K. If all
        them are None, EncoderNet.DEFAULT_PARAMS will be used.

        Args:
            seq_len (int): Number of frames that compose a sequence.
            in_channels (int): Number of channels of images in the input sequence.
            out_channels (int): Number of channels of the output latent encoding.
            hidden_conv_layers (int): Number of hidden convolutional layers (excluding the input
                and the two output layers for mean and variance).
            n_filters (list): List with number of filters for each of the hidden layers.
            kernel_sizes (list): List with kernel sizes for each convolutional layer.
            strides (list): List with strides for each convolutional layer.
            act_func (torch.nn.Module): The activation function to apply after each layer.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()


        self.out_mean = ViT(img_dim=128, in_channels=3, patch_dim=4, num_classes=128*128*48*64, dim=2)
        self.out_logvar = ViT(img_dim=128, in_channels=3, patch_dim=4, num_classes=128*128*48*64, dim=2)

        self.type(dtype)

    def forward(self, x, sample=True):
        """Compute the encoding of the given sequence of images.

        Args:
            x (torch.Tensor): A (batch_size, seq_len * channels, height, width) tensor containing
            the sequence of frames.
            sample (bool): Whether to sample from the encoding distribution or returning the mean.

        Returns:
            A tuple (z, mu, log_var), which are all N x 48 x H x W tensors. z is the latent encoding
            for the given input sequence, while mu and log_var are distribution parameters.
        """
        print(x.shape)
        mean = self.out_mean(x).reshape((x.shape[0], 64, 48, 128, 128))
        if not sample:
            return mean, None, None  # Return None to ensure that they're not used in loss
        log_var = self.out_logvar(x).reshape((x.shape[0], 64, 48, 128, 128))
        stddev = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean)
        z = mean + stddev * epsilon
        return z, mean, log_var