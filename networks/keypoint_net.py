import torch.nn as nn
import torch
from collections import OrderedDict
import pytorch_lightning as pl

class Keypoint_net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params


class Image_encoder_net(nn.Module):
    """Image to encoded feature map"""
    def __init__(self, in_channels=3, initial_num_filters=32, output_map_width=16, layers_per_scale=1, img_width=32):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, initial_num_filters, kernel_size=(3, 3))
        self.layers = OrderedDict()
        for i in range(layers_per_scale):
            self.layers[f'ScalingLayer{i}'] = nn.Conv2d(in_channels, initial_num_filters, kernel_size=(2, 2))

        width = img_width
        num_filters = initial_num_filters

        prev_channels = initial_num_filters
        while width > output_map_width:
            num_filters *= 2
            width //= 2

            # Reduce resolution:
            self.layers['ReduceRes'] = nn.Conv2d(prev_channels, num_filters, kernel_size=(3, 3), stride=2)
            prev_channels = num_filters

            # Apply additional layers:
            for i in range(layers_per_scale):
                self.layers[f'AddConv{i}'] = nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3),
                                                       stride=1)

            self.model = nn.Sequential(self.layers)
            self.out_channels = num_filters

    def forward(self, x):
        x = self.model(x)
        return x

def _get_heatmap_penalty(weight_matrix, factor):
  """L1-loss on mean heatmap activations, to encourage sparsity."""
  weight_shape = weight_matrix.shape.as_list()
  assert len(weight_shape) == 4, weight_shape

  heatmap_mean = torch.mean(weight_matrix, dim=(1, 2))
  penalty = torch.mean(torch.abs(heatmap_mean))
  return penalty * factor


class Images_to_keypoints_net(nn.Module):
    def __init__(self, num_keypoints=10):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.encoder = Image_encoder_net()
        self.features_to_keypoint = \
            nn.Sequential(nn.Softplus(),
                nn.Conv2d(
                    self.encoder.out_channels,
                    self.num_keypoints,
                    kernel_size=1,
                    padding_mode='replicate',

                )
                         )
        # TODO: Add heatmap regularizer penalty
        # TODO: Add coordinates channel



    def forward(self, images):
        encoded = self.encoder(images)
        heatmaps = self.features_to_keypoint(encoded)









