import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from pathlib import Path
import os
import yaml
import numpy as np
from experiments.hgn_combined import HgnTrainer
from networks.resnet import resnet18, resnet18_alternative, resnet18maxpooling,bigger_resnet18
from networks.hamiltonian_net import HamiltonianNet
from networks.hamiltonian_u_net import HamiltonianUNet
from networks.decoder_net import DecoderNet
from utilities.hgn_result import HgnResult
from utilities.integrator import Integrator
from networks.conv_gru import ConvGRU
from networks.discriminator import resnet
from networks.motion_encoder import MotionEncoder
from networks.startframe_encoder import StartFrameEncoder
from utilities.metrics import FVD, calculate_FVD
import random
from utilities.taming_utilities import *


class CodebookEncoder(nn.Module):
    def __init__(self, params, *args, **kwargs,):
        super().__init__()
        self.params = params
        self.batch_size = params['optimization']['batch_size']
        if kwargs.get('load', False):
            pass

        else:
            self.motion_encoder = MotionEncoder(self.params["networks"]['motion_encoder'])

            self.frame_quantize = VectorQuantizer(self.params["frame_book"]["n_embed"], self.params["frame_book"]["embed_dim"], beta=0.25)
            self.frame_quant_conv = torch.nn.Conv2d(self.params["frame_book"]["z_channels"], self.params["frame_book"]["embed_dim"], 1)
            self.frame_post_quant_conv = torch.nn.Conv2d(self.params["frame_book"]["embed_dim"], self.params["frame_book"]["z_channels"], 1)

            self.video_quantize = VectorQuantizer(self.params["video_book"]["n_embed"], self.params["video_book"]["embed_dim"], beta=0.25)
            self.video_quant_conv = torch.nn.Conv2d(self.params["video_book"]["z_channels"], self.params["video_book"]["embed_dim"], 1)
            self.video_post_quant_conv = torch.nn.Conv2d(self.params["video_book"]["embed_dim"], self.params["video_book"]["z_channels"], 1)

            self.start_encoder = StartFrameEncoder(in_channels=3, **self.params["networks"]['start_encoder'])
            self.motion_encoder = MotionEncoder(
                self.params["networks"]['motion_encoder'])

            if 'hamiltonian' in self.params['networks']:
                self.hnn = HamiltonianNet(**self.params['networks']['hamiltonian'])
            elif 'unet' in self.params['networks']:
                self.hnn = HamiltonianUNet(in_channels=2*params["latent_dimensions"]["pq_latent_size"],
                                     **self.params['networks']['unet'])
            elif 'gru' in self.params['networks']:
                self.hnn = ConvGRU(**params['networks']['gru'])
            if params.get('watch_gradients', False):
                wandb.watch(self.hnn, log='all')
            self.integrator = Integrator(delta_t=params["dataset"]["rollout"]["delta_time"],
                                        method=params["integrator"]["method"])
            self.decoder = DecoderNet(
                in_channels=1,
                out_channels=params["dataset"]["rollout"]["n_channels"],
                **params['networks']['decoder'])

    def integrate_step(self, q, p, step_no):
        energy = None
        if 'hamiltonian' in self.params['networks']:
            q, p, energy = self.integrator._lf_step(q=q, p=p, hnn=self.hnn)
        else:
            q, p = self.integrator.step_non_scalar(q=q, p=p, hnn=self.hnn)

        return q, p, energy

    def post_process_rollouts_to_q(self, q, p):
        return q

    def sample(self, p, conditioning_frame, n_steps, prediction):
        pass

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def FRAME_encode(self, x):
        h = self.start_encoder(x) # 64x1x4x4
        h = self.frame_quant_conv(h.squeeze()) # 32x4x4
        #print('FRAME_quantize_conv_result', h.size())
        quant, emb_loss, info = self.frame_quantize(h)
        #print('FRAME_quantized_result', quant.size()) # 32x4x4
        return quant, emb_loss, info

    def FRAME_decode(self, quant, start_frame):
        quant = self.frame_post_quant_conv(quant)
        #print('FRAME_post_quant_conv_result', quant.size()) # 64x4x4
        dec = self.decoder(quant, start_frame) # 3x32x32
        return dec

    def VIDEO_encode(self, x):
        h = self.video_quant_conv(x.squeeze()) # 32x4x4
        #print('VIDEO_quantize_conv_result', h.size())
        quant, emb_loss, info = self.video_quantize(h)
        #print('VIDEO_quantized_result', quant.size()) # 32x4x4
        return quant, emb_loss, info

    def VIDEO_decode(self, quant, start_frame):
        quant = self.video_post_quant_conv(quant)
        #print('VIDEO_post_quant_conv_result', quant.size()) # 64x4x4
        dec = self.decoder(quant, start_frame) # 3x32x32
        #print('VIDEO_decoder_result', dec.size())
        return dec

    #def FRAME_decode_code(self, code_b):
    #    quant_b = self.quantize.embed_code(code_b)
    #    dec = self.FRAME_decode(quant_b)
    #    return dec

    def forward(self, rollout_batch, conditioning_frame, prediction, n_steps):
        all_images = torch.reshape(rollout_batch, (rollout_batch.size(0) * rollout_batch.size(1), 3, 64, 64))
        quant_frame, diff_frame, _ = self.FRAME_encode(all_images)
        image_input_3d = torch.reshape(rollout_batch, (rollout_batch.size(0), rollout_batch.size(1), 3, 64, 64))
        #print('quantized_frames ', image_input_3d.size())
        encoded_video = self.motion_encoder(image_input_3d).squeeze(2)
        #print('# ', encoded_video.size())
        quant_video, diff_video, _ = self.VIDEO_encode(encoded_video)
        #print('quant video', quant_video.size()) # [2, 32, 4, 4]
        quant_video = quant_video.unsqueeze(2)
        #print('unsqueezed', quant_video.size())
        #dec = self.FRAME_decode(quant, conditioning_frame)
        for i in range(n_steps):
            decoded_video = self.decoder(quant_video[:, i], None)  # rollout_batch[:, 0])
            prediction.append_reconstruction(decoded_video)
            #print('#video ', decoded_video.size())

        return prediction, diff_frame, diff_video, None