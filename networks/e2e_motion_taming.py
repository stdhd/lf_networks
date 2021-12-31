import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pathlib import Path
import numpy as np
from networks.hamiltonian_net import HamiltonianNet
from networks.hamiltonian_u_net import HamiltonianUNet
from networks.decoder_net import DecoderNet
from utilities.hgn_result import HgnResult
from utilities.integrator import Integrator
from networks.conv_gru import ConvGRU
from networks.motion_encoder import MotionEncoder
from networks.startframe_encoder import StartFrameEncoder
from utilities.taming_utilities import *
import random


class MotionTamingPredictor(nn.Module):
    def __init__(self, params, *args, **kwargs,):
        super().__init__()
        self.params = params
        self.batch_size = params['optimization']['batch_size']
        if kwargs.get('load', False):
            pass

        else:
            self.start_encoder = StartFrameEncoder(in_channels=3, **self.params["networks"]['start_encoder'])
            self.motion_encoder = MotionEncoder(
                self.params["networks"]['motion_encoder'])

            self.transformer = None
            self.quantizer = VectorQuantizer(self.params["codebook"]["n_embed"],
                                                  self.params["codebook"]["embed_dim"], beta=0.25)
            if params.get('watch_gradients', False):
                wandb.watch(self.hnn, log='all')

            self.decoder = DecoderNet(
                in_channels=params["latent_dimensions"]["pq_latent_size"],
                out_channels=params["dataset"]["rollout"]["n_channels"],
                **params['networks']['decoder'])

            self.quant_conv = torch.nn.Conv2d(32, self.params["codebook"]["embed_dim"], 1)

            self.post_quant_conv = torch.nn.Conv2d(self.params["codebook"]["embed_dim"],
                                                   32, 1)

    def quantize(self, h):
        quant, emb_loss, info = self.quantizer(h)

        return quant, emb_loss

    def encode(self, x):
        h, _, _ = self.motion_encoder(x)
        h = self.quant_conv(h)

        quant, emb_loss, info = self.quantizer(h)
        return quant, emb_loss, info

    def decode(self, quant, conditioning_frame):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, conditioning_frame)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_last_layer(self):
        return self.decoder.out_conv.weight

    def forward(self, motion_context, conditioning_frame, n_steps, predictions, n_context=10, return_histogram=False):
        codebook_diff = 0
        for i in range(n_steps):
            frame, single_codebook_diff, new_usages = self.single_frame_forward(motion_context[:, :n_context], conditioning_frame)
            conditioning_frame = frame
            predictions.append_reconstruction(frame)
            if return_histogram:
                predictions.set_codebook_usages(new_usages)
            codebook_diff += single_codebook_diff
        return predictions, codebook_diff

    def multi_frame_forward(self, rollout_batch, n_frames, n_context, predictions, cond_frame_index=-1, p_use_true_q=-1, return_histogram=False):
        codebook_diff = 0
        dynamic_context = rollout_batch.clone()
        for i in range(n_frames):
            motion_context = dynamic_context[:, i: i+n_context]
            cond_frame = motion_context[:,  cond_frame_index, ...]
            predictions, single_codebook_diff = self.forward(motion_context=motion_context,
                                                                conditioning_frame=cond_frame, n_steps=1,
                                                                predictions=predictions,
                                                                n_context=n_context,
                                                             return_histogram=return_histogram)
            if random.uniform(0., 1.) <= p_use_true_q:
                dynamic_context[:, i + n_context] = rollout_batch[:, i + n_context]
            else:
                dynamic_context[:, i + n_context] = predictions.reconstructed_rollout[:, i]
            codebook_diff += single_codebook_diff
        return predictions, codebook_diff

    def single_frame_forward(self, motion_context, conditioning_frame):
        quant, diff, (perplexity, min_encodings, min_encoding_indices) = self.encode(motion_context)
        usages = torch.unique(min_encoding_indices)
        dec = self.decode(quant, conditioning_frame)
        return dec, diff, usages

    def extrapolate(self, p, conditioning_frame, n_steps, prediction):
        pass

    def sample(self, p, conditioning_frame, n_steps, prediction):
        pass