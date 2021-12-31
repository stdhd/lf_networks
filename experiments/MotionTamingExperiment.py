import numpy as np
import torch
import math
import wandb
from logger.custom_logging import make_video, make_video_second_fix
from losses import RecLoss
from networks.discriminator import resnet
from networks.e2e_motion_taming import MotionTamingPredictor
from experiments.BaseExperiment import BaseExperiment
from utilities.hgn_result import HgnResult
from utilities.taming_utilities import *
from networks.motion_encoder import MotionEncoder
from networks.startframe_encoder import StartFrameEncoder
from utilities.metrics import FVD, calculate_FVD


class MotionTamingTrainer(BaseExperiment):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.sample_prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                                   self.params['logging']['n_samples'], 3, 64, 64)),
                                           device=self.device)

        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size, self.get_n_predictions(), 3, 64, 64)),
                                    device=self.device)
        self.automatic_optimization = True
        self.loss = RecLoss(**params['d_t'])
        self.padding_tensor = torch.zeros(self.batch_size,
                                         self.params['logging'][
                                              'n_samples']+self.params['optimization']['input_frames'],
                                          3, self.params['dataset']['img_size'], self.params['dataset']['img_size'], device=self.device)

    def set_model(self):
        self.model = MotionTamingPredictor(self.params)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size, self.get_n_predictions(), 3, 64, 64)),
                                    device=self.device)
        rollouts = batch['images']
        target_frames = self.get_target_frames(rollouts)

        if self.params['optimization'].get('scheduled_sampling', False):
            p_use_true_q = -1 / (1 + math.exp(-0.5 * self.current_epoch + 5)) + 1
        else:
            p_use_true_q = self.params['optimization'].get('prob_use_true_q', 0.)
        self.log('train/p_use_true_q', p_use_true_q)

        predictions, codebook_diff = self.model.multi_frame_forward(rollout_batch=rollouts,
                                                                         n_frames=self.get_n_predictions(),
                                                                         predictions=self.prediction,
                                                                         n_context=self.params['optimization'][
                                                                             'input_frames'],
                                                                         cond_frame_index=self.get_conditioning_frame_index(),
                                                                p_use_true_q=p_use_true_q, return_histogram=False)
        predicted_frames = predictions.reconstructed_rollout
        if optimizer_idx == 0:
            gen_loss, log_dict_gen = self.loss(codebook_diff, target_frames,
                                               predicted_frames,
                                               optimizer_idx, self.global_step,
                                               last_layer=self.model.get_last_layer(), split="train")
            self.log('train/gen_loss', gen_loss)
            self.log_dict(log_dict_gen)
            return gen_loss

        else:
            disc_loss, log_dict_gen = self.loss(codebook_diff, target_frames,
                                                predicted_frames,
                                                optimizer_idx, self.global_step,
                                                last_layer=self.model.get_last_layer(), split="train")
            self.log('train/gen_loss', disc_loss)
            self.log_dict(log_dict_gen)
            return disc_loss

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
        ]

        optim_g = torch.optim.Adam(params_g)
        optim_disc = torch.optim.Adam(params_disc)

        sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.params["optimization"]['gamma'])
        sched_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=self.params["optimization"]["gamma"])

        return [optim_g, optim_disc], [sched_g, sched_disc]

    def validation_step(self, rollouts, batch_idx):
        prediction, target, hgn_output = self.compute_reconst_kld_errors_per_batch(rollouts)

        if batch_idx <= int(self.params["logging"]["n_fvd_samples"] / rollouts['images'].size(0)):
            self.features_fvd_true_samples.append(rollouts['images'][:, :self.params['logging']['n_samples']].detach().cpu().numpy())
            self.features_fvd_fake_reconstructions.append(prediction.detach().cpu().numpy())
            if self.params['optimization']['training_objective'] == 'posterior':
                self.features_fvd_true_reconstructions.append(target.detach().cpu().numpy())

            elif self.params['optimization']['training_objective'] == 'prior':
                self.features_fvd_true_reconstructions.append(target.detach().cpu().numpy())

        if batch_idx == 0:
            wandb.log({'val/codebook_usage': hgn_output.codebook_usages})
            wandb.log({'val/codebook_unique_codes': hgn_output.get_unique_codebook_ids})
            n_videos_logged = 10
            if self.params['logging']['video']:
                validation_video = make_video(target, prediction,
                                              n_videos_logged, bair=self.params['dataset']['name']=='Bair')

                wandb.log({"video": wandb.Video(validation_video, fps=1, format="gif") })
                continued_prediction_video = self.extra_sample_video(rollouts, n_samples=self.params['logging']['n_samples'])
                wandb.log({"continued_prediction_video": wandb.Video(make_video_second_fix(continued_prediction_video,
                                                                               self.params['optimization']['batch_size'],
                                                                                bair=self.params['dataset']['name'] == 'Bair'), fps=1, format="gif")})

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
        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                            self.get_n_predictions(), 3, 64, 64)),
                                    device=self.device)

        rollouts = rollout_batch['images']
        rollout_len = rollouts.shape[1]
        input_frames = self.params['optimization']['input_frames']
        assert input_frames <= rollout_len, f'Number of input frames {input_frames} needs to be <= total video length {rollout_len}'
        target_frames = self.get_target_frames(rollouts)
        hgn_output, codebook_diff = self.model.multi_frame_forward(rollout_batch=rollouts,
                                                                    n_frames=self.get_n_predictions(),
                                                                    predictions=self.prediction,
                                                                    n_context=self.params['optimization'][
                                                                        'input_frames'],
                                                                   cond_frame_index=self.get_conditioning_frame_index(),
                                                                   return_histogram=True)
        prediction = hgn_output.reconstructed_rollout.detach().cpu()
        return prediction, target_frames, hgn_output

    def extra_sample_video(self, rollout_batch, n_samples):
        self.sample_prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                                   self.params['logging']['n_samples'], 3, 64, 64)),
                                           device=self.device)
        self.padding_tensor = torch.zeros(self.batch_size,
                                         self.params['logging'][
                                              'n_samples']+self.params['optimization']['input_frames'],
                                          3, self.params['dataset']['img_size'], self.params['dataset']['img_size'], device=self.device)
        rollouts = rollout_batch['images']
        self.padding_tensor[:, : self.params['optimization']['input_frames']] = rollouts[:, : self.params['optimization']['input_frames']]
        hgn_output, codebook_diff = self.model.multi_frame_forward(rollout_batch=self.padding_tensor,
                                                                   n_frames=n_samples,
                                                                   predictions=self.sample_prediction,
                                                                   n_context=self.params['optimization']['input_frames'],
                                                                   cond_frame_index=self.get_conditioning_frame_index()
                                                                   )
        return hgn_output.reconstructed_rollout.detach().cpu()



