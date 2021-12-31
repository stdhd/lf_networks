import torch
import torch.nn as nn
from networks.self_attention_cv import ViT


class VisionTransformerNet(nn.Module):
    """Implementation of the encoder-transformer network, that maps the latent space into the
    phase space.
    """

    DEFAULT_PARAMS = {
        'hidden_conv_layers': 2,
        'n_filters': [64, 64, 64],
        'kernel_sizes': [3, 3, 3, 3],
        'strides': [2, 2, 2, 1],
    }

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_conv_layers=None,
                 n_filters=None,
                 kernel_sizes=None,
                 strides=None,
                 act_func=torch.nn.ReLU(),
                 dtype=torch.float,
                 type=None):
        """Instantiate the convolutional layers with the given attributes or using the default
        parameters.

        If K is the total number of layers, then hidden_conv_layers = K - 2. The length of
        n_filters must be K - 1, and that of kernel_sizes and strides must be K. If all
        them are None, TransformerNet.DEFAULT_PARAMS will be used.


        Args:
            in_channels (int): Number of input in_channels.
            out_channels (int): Number of in_channels of q and p
            hidden_conv_layers (int): Number of hidden convolutional layers (excluding the input
                and the two output layers for mean and variance).
            n_filters (list): List with number of filters for each of the hidden layers.
            kernel_sizes (list): List with kernel sizes for each convolutional layer.
            strides (list): List with strides for each convolutional layer.
            act_func (torch.nn.Module): The activation function to apply after each layer.
            dtype (torch.dtype): Type of the weights.
        """
        super().__init__()
        # or
        # input: torch.Size([2, 48, 128, 128])

        self.model = ViT(img_dim=128, in_channels=in_channels, patch_dim=4, num_classes=2 * (out_channels**3), dim=128)
        self.type(dtype)

    def forward(self, x):
        """Transforms the given encoding into two tensors q, p.

        Args:
            x (torch.Tensor): A Tensor of shape (batch_size, channels, H, W).

        Returns:
            Two Tensors q, p corresponding to vectors of abstract positions and momenta.
        """
        x = self.model(x)
        q, p = self.to_phase_space(x)
        return q, p

    @staticmethod
    def to_phase_space(encoding):
        """Takes the encoder-transformer output and returns the q and p tensors.

        Args:
            encoding (torch.Tensor): A tensor of shape (batch_size, channels, ...).

        Returns:
            Two tensors of shape (batch_size, channels/2, ...) resulting from splitting the given
            tensor along the second dimension.
        """
        assert encoding.shape[1] % 2 == 0,\
            'The number of in_channels is odd. Cannot split properly.'
        half_len = int(encoding.shape[1] / 2)
        q = encoding[:, :half_len]
        p = encoding[:, half_len:]
        return q, p
