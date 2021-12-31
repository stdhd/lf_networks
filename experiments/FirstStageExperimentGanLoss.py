import numpy as np
import torch
import wandb
import math
import os

from logger.custom_logging import make_video, make_video_second_fix
from losses import reconstruction_loss, kld_loss, RecLoss, VGGLoss
from networks.e2e_predictor import EndToEndPredictor
from networks.e2e_crossattention_predictor import CrossAttPredictor
from experiments.BaseExperiment import BaseExperiment
from utilities.hgn_result import HgnResult
from utilities.metrics import FVD, calculate_FVD
from PIL import Image
import torch.nn.functional as F
from networks.discriminator import resnet10 as resnet10Discriminator
from einops import rearrange
import torch.nn as nn


class FirstStageTrainerGAN(BaseExperiment):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

        self.sample_prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                                   self.params['logging']['n_samples'], 3, 64, 64)),
                                           q_shape=self.latent_size,
                                           device=self.device)

        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                            self.n_input_frames, 3, 64, 64)),
                                    q_shape=self.latent_size,
                                    device=self.device)

        self.disc_start = self.params['loss_weights'].get('disc_epoch_start', 30)
        self.disc_factor = 1
        self.disc_weight = 1
        self.perc_weight = 1
        self.vgg_loss = VGGLoss().eval()
        self.discriminator = resnet10Discriminator(spatial_size=64, config={'num_classes': 1, 'patch_temp_disc': False})
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)

    @property
    def automatic_optimization(self) -> bool:
        return False

    def set_model(self):
        predictors = {'EndToEnd': EndToEndPredictor, 'CA': CrossAttPredictor}
        self.model = predictors[self.params.get('predictor', 'EndToEnd')](self.params)

    def train_disc(self, x_in_true, x_in_fake, opt):
        if self.current_epoch < self.disc_start:
            return {}, torch.zeros_like(x_in_fake).mean()
        self.discriminator.train()

        opt.zero_grad()

        x_in_true.requires_grad_()

        pred_true = self.discriminator(x_in_true)
        loss_real = self.discriminator.loss(pred_true, real=True)

        self.manual_backward(loss_real)#, opt)

        # fake examples
        pred_fake = self.discriminator(x_in_fake.detach())
        loss_fake = self.discriminator.loss(pred_fake, real=False)
        self.manual_backward(loss_fake)#, opt)

        # optmize parameters
        opt.step()

        loss_disc = ((loss_real + loss_fake) / 2.).item()
        out_dict = {f"train/d_loss": loss_disc, f"train/p_true": torch.sigmoid(pred_true).mean().item(),
                    f"train/p_fake": torch.sigmoid(pred_fake).mean().item()}

        # train generator
        pred_fake = self.discriminator(x_in_fake)

        loss_gen = -torch.mean(pred_fake)

        # loss_fmap = self.disc.fmap_loss(fmap_fake, fmap_true)
        # if self.parallel:
        #     loss_fmap = loss_fmap.cuda(self.devices[0])
        #     loss_gen = loss_gen.cuda(self.devices[0])

        return out_dict, loss_gen  # , loss_fmap

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        (opt_g, opt_d) = self.optimizers()

        rollouts = batch['images']
        rollout_len = rollouts.shape[1]

        assert (self.n_input_frames <= rollout_len)

        input = self.get_input_frames(rollouts)
        target = self.get_target_frames(rollouts)
        cond_frame = self.get_cond_frame(rollouts)
        n_steps = target.size(1)

        p_use_true_q = 0

        self.log('train/p_use_true_q', p_use_true_q)

        self.prediction = HgnResult(batch_shape=torch.Size((input.size(0),
                                                            self.n_input_frames, 3, 64, 64)),
                                    q_shape=self.latent_size,
                                    device=self.device)

        hgn_output, energy_mean = self.model.forward(rollout_batch=input, conditioning_frame=cond_frame,
                                                     prediction=self.prediction,
                                                     n_steps=n_steps, p_use_true_q=p_use_true_q)

        if 'integrator' in self.params:
            self.log_dict({'train/mean_energies': energy_mean,
                           'train/delta_t': self.model.integrator.delta_t})
        prediction = hgn_output.reconstructed_rollout

        rec_loss = torch.abs(target.contiguous() - prediction.contiguous())
        inputs_frames = rearrange(target, 'b n c w h -> (b n) c w h')
        reconstructions_frames = rearrange(prediction, 'b n c w h -> (b n) c w h')
        p_loss = self.vgg_loss(reconstructions_frames.contiguous(), inputs_frames.contiguous())
        # equal weighting of l1 and perceptual loss
        rec_loss = rec_loss + self.perc_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        kl_dict = {}
        if self.params['networks']['motion_encoder'].get('do_reparameterization', True):
            kld = kld_loss(mu=hgn_output.z_mean.detach().cpu(), logvar=hgn_output.z_logvar.detach().cpu(), mean_reduction=True).detach().cpu(
            ).item()
            # normalize by number of frames, channels and pixels per frame
            kld_normalizer = prediction.flatten(1).size(1)
            kld = kld / kld_normalizer
            kl_dict["train/kld"] = float(kld)
            nll_loss += self.params['loss_weights']['kld'] * kld

        d_dict, g_loss = self.train_disc(target, prediction, opt_d)

        # # generator update
        # logits_fake = self.discriminator(rec)
        # g_loss = -torch.mean(logits_fake)

        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, self.disc_weight,
                                             last_layer=list(self.model.decoder.parameters())[-1]) if self.current_epoch >= self.disc_start else 0

        disc_factor = self.adopt_weight(self.disc_factor, self.current_epoch, threshold=self.disc_start)
        loss = nll_loss + d_weight * disc_factor * g_loss

        opt_g.zero_grad()
        self.manual_backward(loss) #, opt_g)
        opt_g.step()

        mean_rec_loss = rec_loss.mean()
        loss_dict = {
                     "train/nll_loss": nll_loss,
                     "train/rec_loss": mean_rec_loss, "train/d_weight": d_weight, "train/disc_factor": disc_factor,
                     "train/g_loss": g_loss, }
        loss_dict.update(d_dict)
        loss_dict.update(kl_dict)

        self.log_dict(loss_dict, logger=True, on_epoch=True, on_step=True)
        # self.logger.experiment.log({k: loss_dict[k].item() if isinstance(loss_dict[k],torch.Tensor) else loss_dict[k] for k in loss_dict})
        self.log("global step", self.global_step)
        self.log("learning rate", opt_g.param_groups[0]["lr"], on_step=True, logger=True)

        # self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)

        self.log("overall_loss", loss, prog_bar=True, logger=False)
        self.log("d_loss", d_dict["train/d_loss"] if "train/d_loss" in d_dict else 0, prog_bar=True, logger=False)
        self.log("nll_loss", nll_loss, prog_bar=True, logger=False)
        self.log("g_loss", g_loss, prog_bar=True, logger=False)
        self.log("logvar", self.logvar.detach(), prog_bar=True, logger=True)
        self.log("rec_loss", mean_rec_loss, prog_bar=True, logger=False)

        return loss_dict, batch_idx

    def calculate_adaptive_weight(self, nll_loss, g_loss, discriminator_weight, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def adopt_weight(self, weight, epoch, threshold=0, value=0.):
        if epoch < threshold:
            weight = value
        return weight

    def get_last_layer(self):
        return list(self.model.decoder.parameters())[-1]

    def configure_optimizers(self):
        params = self.params
        params_g = [
            {
                'params': self.model.start_encoder.parameters(),
                'lr': params["optimization"]["frame_encoder_lr"],
                'name': 'start frame encoder',
                'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.motion_encoder.parameters(),
                'lr': params["optimization"]["motion_encoder_lr"],
                'name': 'motion encoder',
                'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.hnn.parameters(),
                'lr': params["optimization"]["hnn_lr"],
                'name': 'HNN or UNET',
                'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.decoder.parameters(),
                'lr': params["optimization"]["decoder_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'decoder'
            },
            {
                'params': self.logvar,
                'lr': params["optimization"]["decoder_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'logvar'
            },
        ]

        if 'integrator' in params:
            params_g.append({
                'params': self.model.integrator.parameters(),
                'lr': params["optimization"]["integrator_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'integrator'
            },)

        if 'context_encoder' in params['networks']:
            params_g.append({
                'params': self.model.state_to_context_encoder.parameters(),
                'lr': params["optimization"]["hnn_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'state_to_context_encoder'
            }

            )

        params_d = [
                {
                    'params': self.discriminator.parameters(),
                    'lr': params["optimization"]["codebook_lr"],
                    'weight_decay': params["optimization"].get("weight_decay", 0),
                    'name': 'post_quant_conv'
                }, ]






        optim_g = torch.optim.Adam(params_g)
        optim_d = torch.optim.Adam(params_d)


        return [optim_g, optim_d] , []


    def get_kl_scheduled(self):
        return float(self.current_epoch * self.params['loss_weights']['kld']) \
            if self.params['loss_weights'].get('use_kld_scheduling', False) \
            else self.params['loss_weights']['kld']

    def compute_reconst_kld_errors_per_batch(self, rollout_batch):
        """Computes reconstruction error and KL divergence."""

        # Move to device and change dtype
        rollouts = rollout_batch['images']
        rollout_len = rollouts.shape[1]
        assert self.n_input_frames <= rollout_len, f'Number of input frames {self.n_input_frames} needs to be <= total video length {rollout_len}'

        input = self.get_input_frames(rollouts)
        target = self.get_target_frames(rollouts)
        cond_frame = self.get_cond_frame(rollouts)

        n_steps = target.size(1)
        self.prediction = HgnResult(batch_shape=torch.Size((rollouts.size(0),
                                                            self.n_input_frames, 3, 64, 64)),
                                    q_shape=(self.params['latent_dimensions']['pq_latent_size'],
                                             self.params['latent_dimensions']['pq_latent_dim'],
                                             self.params['latent_dimensions']['pq_latent_dim']),
                                    device=self.device, log_states=True)

        hgn_output, energy_mean = self.model.forward(rollout_batch=input, conditioning_frame=cond_frame, n_steps=n_steps, prediction=self.prediction)

        prediction_detached = hgn_output.reconstructed_rollout.detach().cpu()

        error = reconstruction_loss(
            target=target.detach().cpu(),
            prediction=prediction_detached, mean_reduction=True).detach().cpu(
        ).detach().cpu().item()

        err_dict = {

            "val/reconstruction_loss": float(error),
        }

        if self.params['networks']['motion_encoder'].get('do_reparameterization', True):
            kld = kld_loss(mu=hgn_output.z_mean.detach().cpu(), logvar=hgn_output.z_logvar.detach().cpu(), mean_reduction=True).detach().cpu(
            ).item()
            # normalize by number of frames, channels and pixels per frame
            kld_normalizer = prediction_detached.flatten(1).size(1)
            kld = kld / kld_normalizer

            err_dict["val/kld"] = float(kld)

        return err_dict, hgn_output
#err_dict, prediction, hgn_output.reconstructed_rollout.detach()







