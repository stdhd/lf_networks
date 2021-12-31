import numpy as np
import torch
import wandb

from logger.custom_logging import make_video, make_video_second_fix
from losses import reconstruction_loss, kld_loss, RecLoss
from networks.discriminator import resnet
from networks.codebookEncoder import CodebookEncoder
from experiments.BaseExperiment import BaseExperiment
from utilities.hgn_result import HgnResult
from utilities.taming_utilities import *
from networks.motion_encoder import MotionEncoder
from networks.startframe_encoder import StartFrameEncoder
from utilities.metrics import FVD, calculate_FVD


class TamingTrainer(BaseExperiment):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.sample_prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                                   self.params['logging']['n_samples'], 3, 64, 64)),
                                           device=self.device)

        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                            self.params['optimization']['input_frames'], 3, 64, 64)),
                                    device=self.device)
        self.automatic_optimization = True
        if self.params['d_t']['use']:
            self.loss = RecWithDiscriminatorLoss(disc_start=self.params['d_t']['disc_start'], params=params)
        else:
            self.loss = RecLossWithCodebook(params=params)

    def set_model(self):
        self.model = CodebookEncoder(self.params)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        rollouts = batch['images']
        rollout_len = rollouts.shape[1]
        input_frames = self.params['optimization']['input_frames']

        assert (input_frames <= rollout_len)
        input = rollouts[:, 1:input_frames + 1]

        if self.params['optimization']['training_objective'] == 'posterior':
            target = rollouts[:, input_frames + 1:]
            cond_frame = target[:, 0]

        elif self.params['optimization']['training_objective'] == 'prior':
            target = rollouts[:, 1:input_frames + 1]
            cond_frame = rollouts[:, 0]

        else:
            raise RuntimeError('parameter training_objective is not properly defined')
        n_steps = target.size(1)
        self.prediction = HgnResult(batch_shape=torch.Size((input.size(0),
                                                            self.params['optimization']['input_frames'], 3, 64, 64)),
                                    device=self.device)

        hgn_output, frame_codebook_loss, video_codebook_loss, energy_mean = self.model.forward(rollout_batch=input, conditioning_frame=cond_frame,
                                                     prediction=self.prediction,
                                                     n_steps=n_steps)

        prediction = hgn_output.reconstructed_rollout

        if optimizer_idx == 0:
            gen_loss, log_dict_gen = self.loss((frame_codebook_loss, video_codebook_loss), target,
                                               prediction,
                                               optimizer_idx, self.global_step,
                                               last_layer=None, split="train")
            self.log('train/gen_loss', gen_loss)
            self.log_dict(log_dict_gen)
            return gen_loss

    def configure_optimizers(self):
        # Define optimization modules
        params = self.params
        params_g = [
            {
                'params': self.model.parameters(),
                'lr': params["optimization"]["frame_encoder_lr"],
                'name': 'start frame encoder',
                'weight_decay': params["optimization"].get("weight_decay", 0),
            },
        ]

        params_disc = [
            {
                'params': self.loss.discriminator.parameters(),
                'lr': params["optimization"]["disc_t_lr"],
                'name': 'discriminator'
            }
        ] if self.params['d_t'].get('use', False) else []

        optim_g = torch.optim.Adam(params_g)
        if self.params['d_t'].get('use', False):
            optim_disc = torch.optim.Adam(params_disc)  # TODO: Add weight decay value

        sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.params["optimization"]['gamma'])
        if self.params['d_t'].get('use', False):
            sched_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=self.params["optimization"]["gamma"])

        return [optim_g, optim_disc] if self.params['d_t'].get('use', False) else [optim_g], \
               [sched_g, sched_disc] if self.params['d_t'].get('use', False) else [sched_g]

    def validation_step(self, rollouts, batch_idx):
        if 'hamiltonian' in self.params['networks']:
            torch.set_grad_enabled(True)
        prediction = self.compute_reconst_kld_errors_per_batch(rollouts)

        if batch_idx <= int(self.params["logging"]["n_fvd_samples"] / rollouts['images'].size(0)):
            self.features_fvd_true_samples.append(rollouts['images'][:, :self.params['logging']['n_samples']].detach().cpu().numpy())
            self.features_fvd_fake_reconstructions.append(prediction.detach().cpu().numpy())
            input_frames = self.params['optimization']['input_frames']
            if self.params['optimization']['training_objective'] == 'posterior':
                self.features_fvd_true_reconstructions.append(rollouts['images'][:, input_frames:].detach().cpu().numpy())

            elif self.params['optimization']['training_objective'] == 'prior':
                self.features_fvd_true_reconstructions.append(rollouts['images'][:, :input_frames].detach().cpu().numpy())

        if batch_idx == 0:
            n_videos_logged = 10
            predicted = prediction
            input_frames = self.params['optimization']['input_frames']
            if self.params['optimization']['training_objective'] == 'posterior':
                target = rollouts['images'][:, input_frames:]
            elif self.params['optimization']['training_objective'] == 'prior':
                target = rollouts['images'][:, :input_frames]
            else:
                raise RuntimeError('parameter training_objective is not properly defined')

            if self.params['logging']['video']:
                validation_video = make_video(target, predicted,
                                              n_videos_logged, bair=self.params['dataset']['name']=='Bair')

                wandb.log({"video": wandb.Video(validation_video, fps=1, format="gif"),
                           # "last_frame_before_prediction": wandb.Image(outs[0][1]['images'][:, input_frames].clone().cpu())

                           })

    def validation_epoch_end(self, outs):
        self.FVD.i3d.eval()
        features_fake_reconstruction = torch.from_numpy(np.concatenate(self.features_fvd_fake_reconstructions, axis=0))
        features_true_reconstruction = torch.from_numpy(np.concatenate(self.features_fvd_true_reconstructions, axis=0))

        fvd_score_reconstruction = calculate_FVD(self.FVD.i3d, features_fake_reconstruction, features_true_reconstruction,
                                          batch_size=self.params["logging"]["bs_i3d"], cuda=True)
        self.features_fvd_fake_samples.clear()
        self.features_fvd_true_samples.clear()

        self.features_fvd_fake_reconstructions.clear()
        self.features_fvd_true_reconstructions.clear()

        self.log_dict({'val/fvd_reconstruction': float(fvd_score_reconstruction)})
        np.random.seed(self.current_epoch)

    def compute_reconst_kld_errors_per_batch(self, rollout_batch):
        """Computes reconstruction error and KL divergence."""

        # Move to device and change dtype
        rollouts = rollout_batch['images']
        rollout_len = rollouts.shape[1]
        input_frames = self.params['optimization']['input_frames']
        assert input_frames <= rollout_len, f'Number of input frames {input_frames} needs to be <= total video length {rollout_len}'

        input = rollouts[:, :input_frames]
        if self.params['optimization']['training_objective'] == 'posterior':
            target = rollouts[:, input_frames:]
            cond_frame = target[:, 0]

        elif self.params['optimization']['training_objective'] == 'prior':
            target = rollouts[:, :input_frames]
            cond_frame = rollouts[:, 0]

        else:
            raise RuntimeError('parameter training_objective is not properly defined')
        n_steps = target.size(1)
        self.prediction = HgnResult(batch_shape=torch.Size((rollouts.size(0),
                                                            self.params['optimization']['input_frames'], 3, 64, 64)),
                                    device=self.device)
        # print('sizes ', input.size(), cond_frame.size())

        hgn_output, frame_codebook_loss, video_codebook_loss, energy_mean= self.model.forward(rollout_batch=input, conditioning_frame=cond_frame,
                                                     n_steps=n_steps, prediction=self.prediction)

        prediction = hgn_output.reconstructed_rollout.detach().cpu()

        return prediction


