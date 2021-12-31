import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from logger.custom_logging import make_video, make_video_second_fix, make_pq_diagram
from losses import reconstruction_loss, kld_loss, VGGLoss
from networks.discriminator import resnet
from networks.e2e_predictor import EndToEndPredictor
from utilities.hgn_result import HgnResult
from utilities.metrics import FVD, calculate_FVD, LPIPS
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageOps
import torch.nn as nn
import os



class BaseExperiment(pl.LightningModule):

    def __init__(self, params, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.batch_size = params['optimization']['batch_size']
        self.set_model()
        self.FVD = FVD(n_samples=params['logging']['n_fvd_samples'])
        #self.LPIPS = LPIPS()
        self.features_fvd_true_samples = []
        self.features_fvd_fake_samples = []
        self.features_fvd_true_reconstructions = []
        self.features_fvd_fake_reconstructions = []
        self.n_input_frames = params['optimization']['input_frames']
        self.sequence_length = params['dataset']['sequence_length']
        self.save_test_dir = ''
        self.ref_dir = os.path.join(self.save_test_dir, 'references')
        self.pred_dir = os.path.join(self.save_test_dir, 'predictions')
        self.save_test_n_videos = params['logging'].get('save_test_n_videos', 100)
        self.latent_size = (self.params['latent_dimensions']['pq_latent_size'],
                                self.params['latent_dimensions']['pq_latent_dim'],
                                self.params['latent_dimensions']['pq_latent_dim'])

        self.p_latent_sample = torch.randn(
            (self.params['optimization']['batch_size'], self.params['latent_dimensions']['pq_latent_size'],
             self.params['latent_dimensions']['pq_latent_dim'],
             self.params['latent_dimensions']['pq_latent_dim']), device=self.device)

    def set_model(self):
        raise NotImplementedError('subclasses must override this method')

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError('subclasses must override this method')

    def configure_optimizers(self):
        raise NotImplementedError('subclasses must override this method')

    def get_kl_scheduled(self):
        return float(self.current_epoch * self.params['loss_weights']['kld']) \
            if self.params['loss_weights'].get('use_kld_scheduling', False) \
            else self.params['loss_weights']['kld']

    def validation_step(self, rollouts, batch_idx):
        if 'hamiltonian' in self.params['networks']:
            torch.set_grad_enabled(True)
        err_dict, hgn_output = self.compute_reconst_kld_errors_per_batch(rollouts)
        prediction = hgn_output.reconstructed_rollout.detach().cpu()
        self.log_dict(err_dict)
        if batch_idx <= int(self.params["logging"]["n_fvd_samples"] / rollouts['images'].size(0)):
            self.p_latent_sample = torch.randn((rollouts['images'].size(0),
                                            self.params['latent_dimensions']['pq_latent_size'],
                                                self.params['latent_dimensions']['pq_latent_dim'],
                                                self.params['latent_dimensions']['pq_latent_dim']), device=self.device
                                               , requires_grad=True)
            self.sample_prediction = HgnResult(batch_shape=torch.Size((rollouts['images'].size(0),
                                                                   self.params['logging']['n_samples'], 3, 64, 64)), device=self.device)
            X_hat_sample = self.model.sample(self.p_latent_sample, rollouts['images'][:, 0],
                                              n_steps=self.params['logging']['n_samples'], prediction=self.sample_prediction).reconstructed_rollout.detach().cpu().numpy()

            self.features_fvd_fake_samples.append(X_hat_sample)
            self.features_fvd_true_samples.append(rollouts['images'][:, :self.params['logging']['n_samples']].detach().cpu().numpy())
            self.features_fvd_fake_reconstructions.append(prediction.detach().cpu().numpy())
            input_frames = self.params['optimization']['input_frames']
            if self.params['optimization']['training_objective'] == 'posterior':
                self.features_fvd_true_reconstructions.append(rollouts['images'][:, input_frames:].detach().cpu().numpy())
                #self.LPIPS.update(prediction_gpu.detach().reshape(-1, 3, 64, 64),
                #                  rollouts['images'][:, :input_frames].detach().reshape(-1, 3, 64, 64))

            elif self.params['optimization']['training_objective'] == 'prior':
                self.features_fvd_true_reconstructions.append(rollouts['images'][:, :input_frames].detach().cpu().numpy())
                #self.LPIPS.update(prediction_gpu.detach().reshape(-1, 3, 64, 64), rollouts['images'][:, :input_frames].detach().reshape(-1, 3, 64, 64))

        if batch_idx == 0:
            phase_plot = make_pq_diagram(p=hgn_output.p, q=hgn_output.q)
            wandb.log({'val/phase_space': phase_plot})
            self.p_latent_sample = torch.randn(
                (rollouts['images'].size(0), self.params['latent_dimensions']['pq_latent_size'],
                 self.params['latent_dimensions']['pq_latent_dim']
                 , self.params['latent_dimensions']['pq_latent_dim']),
                device=self.device, requires_grad=True)
            conditioning_frame = rollouts['images'][: self.params['optimization']['batch_size'], 0]
            self.sample_prediction = HgnResult(batch_shape=torch.Size((rollouts['images'].size(0),
                                                                       self.params['logging']['n_samples'], 3, 64, 64)),
                                               device=self.device)
            samples = self.model.sample(self.p_latent_sample, conditioning_frame,
                                        n_steps=self.params.get('logging', {}).get('n_samples', 5),
                                        prediction=self.sample_prediction).reconstructed_rollout.detach()
            sampling_video = make_video_second_fix(samples, self.params['optimization']['batch_size'],
                                                   bair=self.params['dataset']['name'] == 'Bair')
            wandb.log(
                {
                    'val/sampling/rollouts': wandb.Video(sampling_video, fps=1, format="gif"),
                    'val/sampling/conditioning_frame': wandb.Image(conditioning_frame),
                }
            )

            n_videos_logged = 10
            predicted = prediction
            target = self.get_target_frames(rollouts['images'])

            if self.params['logging']['video']:
                validation_video = make_video(target, predicted,
                                              n_videos_logged, bair=self.params['dataset']['name']=='Bair')

                wandb.log({"video": wandb.Video(validation_video, fps=1, format="gif"),
                           # "last_frame_before_prediction": wandb.Image(outs[0][1]['images'][:, input_frames].clone().cpu())

                           })

    def validation_epoch_end(self, outs):
        self.FVD.i3d.eval()

        features_fake_samples = torch.from_numpy(np.concatenate(self.features_fvd_fake_samples, axis=0))
        features_true_samples = torch.from_numpy(np.concatenate(self.features_fvd_true_samples, axis=0))

        features_fake_reconstruction = torch.from_numpy(np.concatenate(self.features_fvd_fake_reconstructions, axis=0))
        features_true_reconstruction = torch.from_numpy(np.concatenate(self.features_fvd_true_reconstructions, axis=0))
        fvd_score_samples = calculate_FVD(self.FVD.i3d, features_fake_samples, features_true_samples,
                                  batch_size=self.params["logging"]["bs_i3d"], cuda=True)
        fvd_score_reconstruction = calculate_FVD(self.FVD.i3d, features_fake_reconstruction, features_true_reconstruction,
                                          batch_size=self.params["logging"]["bs_i3d"], cuda=True)
        self.features_fvd_fake_samples.clear()
        self.features_fvd_true_samples.clear()

        self.features_fvd_fake_reconstructions.clear()
        self.features_fvd_true_reconstructions.clear()

        self.log_dict({'val/fvd_sampling': float(fvd_score_samples)})
        self.log_dict({'val/fvd_reconstruction': float(fvd_score_reconstruction)})
        #self.log_dict({'val/lpips_reconstruction': float(self.LPIPS.compute())})
        np.random.seed(self.current_epoch)

    def compute_reconst_kld_errors_per_batch(self, rollout_batch):
        pass

    def get_checkpoint_monitoring(self, logging_root):
        return ModelCheckpoint(dirpath=logging_root,
                               monitor='val/fvd_reconstruction',
                               mode='min',
                               auto_insert_metric_name=True)


    def get_target_frames(self, batch):
        if self.params['optimization']['training_objective'] == 'posterior':
            return batch[:, self.n_input_frames :]
        elif self.params['optimization']['training_objective'] == 'prior':
            return batch[:, 1:1+self.n_input_frames]

    def get_input_frames(self, batch):
        if self.params['optimization']['training_objective'] == 'posterior':
            return batch[:, :self.n_input_frames]
        elif self.params['optimization']['training_objective'] == 'prior':
            assert batch.size(1) >= self.n_input_frames
            #return batch[:, 1:1+self.n_input_frames]
            return batch[:, :self.n_input_frames + 1]

    def get_conditioning_frame_index(self):
        if self.params['optimization']['training_objective'] == 'posterior':
            return -1
        elif self.params['optimization']['training_objective'] == 'prior':
            return 0

    def get_cond_frame(self, batch):
        if self.params['optimization']['training_objective'] == 'posterior':
            return batch[:, self.n_input_frames-1]
        elif self.params['optimization']['training_objective'] == 'prior':
            return batch[:, 0]

    def get_n_predictions(self):
        if self.params['optimization']['training_objective'] == 'posterior':
            return self.sequence_length - self.n_input_frames
        elif self.params['optimization']['training_objective'] == 'prior':
            return self.n_input_frames

    def test_step(self, batch, batch_idx):
        self.test_reconstruction(batch, batch_idx)

    def init_test_setting(self):
        pass

    def on_test_model_eval(self):
        self.save_test_dir = '/home/sd/Documents/thesis/fvd/evaluations'
        self.ref_dir = os.path.join(self.save_test_dir, 'references')
        self.pred_dir = os.path.join(self.save_test_dir, 'predictions')
        self.diff_dir = os.path.join(self.save_test_dir, 'diffs')
        os.mkdir(self.diff_dir)
        os.mkdir(self.ref_dir)
        os.mkdir(self.pred_dir)

    def test_reconstruction(self, batch, batch_idx):
        if 'hamiltonian' in self.params['networks']:
            torch.set_grad_enabled(True)
        err_dict, hgn_output = self.compute_reconst_kld_errors_per_batch(batch)
        prediction = hgn_output.reconstructed_rollout.detach().cpu()
        start_frames = np.rollaxis(self.get_cond_frame(batch['images'].cpu().numpy()), 1, 4)

        for bi in range(prediction.size(0)):
            video_dir = os.path.join(self.pred_dir, str(batch_idx) + '_' + str(bi))
            references_dir = os.path.join(self.ref_dir, str(batch_idx) + '_' + str(bi))
            #startframe_pred_dir = os.path.join(video_dir, str(0))
            #startframe_ref_dir = os.path.join(references_dir, str(0))

            diff_dir = os.path.join(self.diff_dir, str(batch_idx) + '_' + str(bi))
            os.makedirs(video_dir)
            os.makedirs(references_dir)
            os.makedirs(diff_dir)

            #startframe = start_frames[bi]

            #with Image.fromarray((startframe * 255).astype(np.uint8)) as im:
            #    im.save(f'{startframe_pred_dir}.png')
            #    im.save(f'{startframe_ref_dir}.png')

            #with Image.fromarray((startframe * 255).astype(np.uint8)) as im:
            #    im.save(f'{startframe_ref_dir}.png')

            for vi in range(prediction.size(1)):
                frame_dir = str(vi)
                frame = np.rollaxis(prediction[bi, vi].numpy(), 0, 3)
                with Image.fromarray((frame * 255).astype(np.uint8)) as im:
                    im.save(f'{os.path.join(video_dir, frame_dir)}.png')

                ref = np.rollaxis(batch['images'][bi, vi].cpu().numpy(), 0, 3)
                with Image.fromarray((ref * 255).astype(np.uint8)) as im:
                    im.save(f'{os.path.join(references_dir, frame_dir)}.png')

                diff = np.abs((start_frames[bi] - frame))
                with Image.fromarray((diff * 255).astype(np.uint8)) as im:
                    im = ImageOps.grayscale(im)
                    im.save(f'{os.path.join(diff_dir, frame_dir)}.png')
