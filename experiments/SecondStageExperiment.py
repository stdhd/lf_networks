from logger.custom_logging import make_video_second_fix
from .utils import *
from .FirstStageExperiment import FirstStageTrainer
from networks.inn.loss import FlowLoss
import torch
from utilities.hgn_result import HgnResult
from networks.inn.inn import *
from experiments.BaseExperiment import BaseExperiment
from networks.inn.inn import UnconditionalFlow, ConditionalConvFlow
from networks.startframe_encoder import StartFrameEncoder
from pytorch_lightning.callbacks import ModelCheckpoint
from networks.startframe_encoder import get_encoder
import os
import yaml
import wandb
from utilities.metrics import FVD, calculate_FVD
from PIL import Image, ImageOps
import lpips
from einops import rearrange, reduce, repeat
import torch.nn.functional as f




def _read_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config is None:
        config = {}
    return config


class SecondStageTrainer(BaseExperiment):
    def __init__(self, params, **kwargs):
        print(params['optimization']['batch_size'])

        super().__init__(params, **kwargs)
        print(params['optimization']['batch_size'])

        self.lpips = lpips.LPIPS(net='alex')

        self.z_dimensions = (params['optimization']['batch_size'], 32, 8, 8) # (self.params['optimization']['batch_size'], 2048)

        self.register_buffer('z_sample', torch.randn(self.z_dimensions), persistent=False)
        print('REGISTER buffer ', self.z_dimensions)
        self.loss_func = FlowLoss()

        if params['networks']['inn'].get('type', '') == 'macow':
            self.inn = SupervisedMacowTransformer(params['networks']['inn'])
        elif params['networks']['inn'].get('type', '') == 'UnsupervisedMaCowTransformer3':
            self.inn = UnsupervisedMaCowTransformer3(params['networks']['inn'])
        elif params['networks']['inn'].get('type', '') == 'UnsupervisedConvTransformer':
            self.inn = UnsupervisedConvTransformer(params['networks']['inn'])
        elif  params['networks']['inn'].get('type', '') == 'UnconditionalMaCowFlow':
            self.inn = UnconditionalMaCowFlow(params['networks']['inn'])
        elif params['networks']['inn'].get('type', '') == 'SupervisedMacowTransformer':
            params["networks"]["inn"]["flow_mid_channels"] = int(params["networks"]["inn"]["flow_mid_channels_factor"] * \
                                                                 params["networks"]["inn"]["flow_in_channels"])
            self.inn = SupervisedMacowTransformer(params['networks']['inn'])
        elif params['networks']['inn'].get('type', '') == 'ConditionalConvFlow':
            self.inn = ConditionalConvFlow(**params['networks']['inn'])
        elif params['networks']['inn'].get('conditional', True): #conditional
            self.inn = ConditionalConvFlow(in_channels=32,
                                           embedding_dim=32, # self.params['latent_dimensions']['pq_latent_size'],
                                         **self.params['networks']['inn'])
        else:
            self.inn = UnconditionalFlow(in_channels=2048,  # self.params['latent_dimensions']['pq_latent_size'],
                                         **self.params['networks']['inn'])
        f_params = _read_config(self.params['first_stage_config'])
        f_params['optimization']['batch_size'] = params['optimization']['batch_size']
        print(params['optimization']['batch_size'])

        from experiments import __experiments__
       # first_stage_exp = select_experiment(f_params)
        self.first_stage_model = __experiments__[f_params['experiment']](params=f_params).load_from_checkpoint(
            checkpoint_path=self.params['first_stage_checkpoint'])
        self.first_stage_model.freeze()
        if self.params['networks']['condition_encoder'].get('use_start_frame_encoder', False):
            self.condition_encoder = self.first_stage_model.model.start_encoder
            print('use pretrained condition encoder')
        else:
            self.condition_encoder = get_encoder(in_channels=3, **self.params["networks"]['condition_encoder'])

        if self.params['networks']['condition_encoder'].get('freeze', True):
            for param in self.condition_encoder.parameters():
                param.requires_grad = False
        else:
            self.condition_encoder.train()

    def kld(mu, logvar, mean_reduction=True):
        mu = mu.flatten(1)
        logvar = logvar.flatten(1)
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if mean_reduction:
            return torch.mean(kld_per_sample, dim=0)
        else:
            return kld_per_sample

    def forward_flow(self, input, condition):
        flowInput = self.get_flow_input(input)
        if self.params['networks']['inn']['conditional']:
            condition = self.condition_encoder(condition)
            out, logdet = self.inn.forward(flowInput.detach(),  # .reshape(-1, 2048),
                                           condition,
                                           reverse=False)
        else:
            out, logdet = self.inn.forward(flowInput.detach(),
                                           reverse=False)

        # assert flowInput.size() == self.z_dimensions, 'Check if z dimensions match fails'
        # self.inn.forward(flowInput.detach().reshape(-1, 128, 4, 4), reverse=False)
        return out, logdet

    def training_step(self, batch, batch_idx):
        condition = self.get_cond_frame(batch['images'])
        flow_input = self.get_input_frames(batch['images'])
        out, logdet = self.forward_flow(flow_input, condition=condition)

        loss, loss_dict = self.loss_func(out, logdet)
        self.log_dict(loss_dict)
        return loss

    @no_grad
    def _get_sample_flow_reverse(self, condition=None):
        # Sample from the standard gaussian and transform to latent motion variable
        #z_n = torch.randn(self.z_dimensions)# self.z_sample
        if self.params['networks']['inn']['conditional']:
            z_latent = self.inn.forward(self.z_sample, condition, reverse=True)
        else:
            z_latent = self.inn.forward(self.z_sample, reverse=True)
        self.z_sample = torch.randn(self.z_dimensions, device=self.z_sample.device)
        #self.z_n = torch.randn(self.z_dimensions)
        return z_latent

    @no_grad
    def get_sample_video(self, cond_frame, n_steps):
        # Predict video given conditioning frame
        condition = self.condition_encoder(cond_frame)
        sampled_motion = self._get_sample_flow_reverse(condition)
        motion = sampled_motion.clone().detach()#.requires_grad_(True)

        self.prediction = HgnResult(batch_shape=torch.Size(((self.params['optimization']['batch_size']),
                                                            n_steps, 3, 64,
                                                            64)),
                                    device=self.device)
        prediction, energy_per_step = self.first_stage_model.model.forward(conditioning_frame=cond_frame,
                                                                           prediction=self.prediction,
                                                                           n_steps=n_steps,
                                                                           initial_p=motion.reshape(-1, 32, 8, 8))

        return prediction, energy_per_step

    def on_test_start(self) -> None:
        self.lpips_array = np.zeros((256*1, 30))
        self.mse_array = np.zeros((256*1, 30))
        self.abs_diff_array = np.zeros((256 * 1, 30))

    def test_step(self, batch, batch_idx):
        n_iterations = 1
        n_steps = 30
        n_variations = 1
        save = True
        cond_frame = self.get_cond_frame(batch['images'])
        for i in range(n_iterations):
            sampled_videos = []
            for variation in range(n_variations):
                sampled_video, _ = self.get_sample_video(cond_frame, n_steps=n_steps)

                sampled_videos.append(sampled_video.reconstructed_rollout.cpu())

                # rearrange elements according to the pattern
                preds = rearrange(sampled_video.reconstructed_rollout, 'b n c w h -> (b n) c w h')
                #refs = rearrange(self.get_input_frames(batch['images']), 'b n c w h -> (b n) c w h')
                refs = rearrange(batch['images'], 'b n c w h -> (b n) c w h')

                lp = self.lpips.forward(preds, refs).squeeze()
                lpips = rearrange(lp, '(b n) -> b n ', n=n_steps)

                self.lpips_array[2 * batch_idx : 2 * batch_idx + 2, :] = lpips.cpu().numpy()
                self.mse_array[2 * batch_idx : 2 * batch_idx + 2, :] = f.mse_loss(preds.cpu(), refs.cpu())

                #exit()
                mse_diff = torch.sum(torch.square(preds.cpu() - refs.cpu()).view(preds.size(0), -1), dim=-1)
                mse_diff = rearrange(mse_diff, '(b n) -> b n ', n=n_steps)
                self.abs_diff_array[2 * batch_idx : 2 * batch_idx + 2, :] = mse_diff
                #print(torch.abs(preds.cpu() - refs.cpu().view(0,1, -1)).size())
                #result_dict['lpips'] = self.lpips.forward(preds, refs)
                #print(torch.mean(result_dict['lpips']))
            if save:
                start_frames = np.rollaxis(self.get_cond_frame(batch['images'].cpu().numpy()), 1, 4)

                for bi in range(batch['images'].size(0)):

                    for variationi in range(n_variations):
                        video_dir = os.path.join(self.pred_dir, str(batch_idx) + '_' + str(bi) + 'variation'+ str(variationi))
                        references_dir = os.path.join(self.ref_dir, str(batch_idx) + '_' + str(bi) + 'variation'+ str(variationi))

                        startframe_pred_dir = os.path.join(video_dir, str(0))
                        startframe_ref_dir = os.path.join(references_dir, str(0))

                        os.makedirs(video_dir)
                        os.makedirs(references_dir)

                        startframe = start_frames[bi]

                        with Image.fromarray((startframe * 255).astype(np.uint8)) as im:
                            im.save(f'{startframe_pred_dir}.png')
                            im.save(f'{startframe_ref_dir}.png')
                        sampled_video = sampled_videos[variationi]
                        for vi in range(sampled_video.size(1)):
                            frame_dir = str(vi + 1)
                            frame = np.rollaxis(sampled_video[bi, vi].numpy(), 0, 3)
                            with Image.fromarray((frame * 255).astype(np.uint8)) as im:
                                im.save(f'{os.path.join(video_dir, frame_dir)}.png')

                            ref = np.rollaxis(batch['images'][bi, vi].cpu().numpy(), 0, 3)
                            with Image.fromarray((ref * 255).astype(np.uint8)) as im:
                                im.save(f'{os.path.join(references_dir, frame_dir)}.png')

    def on_test_end(self) -> None:
        print(self.lpips_array)
        print('lpips_std ', np.std(self.lpips_array))
        print('lpips_mean', np.mean(self.lpips_array))
        print('lpips_mean', np.mean(self.lpips_array, axis=0))
        print('lpips_std', np.std(self.lpips_array, axis=0))

        print('mse_std ', np.std(self.mse_array))
        print('mse_mean', np.mean(self.mse_array))
        print('mse_mean', np.mean(self.mse_array, axis=0))
        print('mse_std', np.std(self.mse_array, axis=0))
        np.save(os.path.join(self.pred_dir, 'lpips.npy'), self.lpips_array)
        np.save(os.path.join(self.pred_dir, 'mse.npy'), self.mse_array)
        np.save(os.path.join(self.pred_dir, 'abs_diff_array.npy'), self.abs_diff_array)


    @no_grad
    def get_flow_input(self, input):
        motion, mu, logvar = self.first_stage_model.model.motion_encoder.forward(input)
        return motion

    def configure_optimizers(self):
        # Define optimization modules
        params = self.params
        params_inn = [
            {
                'params': self.inn.parameters(),
                'lr': params["optimization"]["inn_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'inn'
            },


        ]

        if not (self.params['networks']['condition_encoder'].get('freeze', True)):
            params_inn.append({
                'params': self.condition_encoder.parameters(),
                'lr': params["optimization"]["inn_lr"],
                'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'inn'
            })

        optim_inn = torch.optim.Adam(params_inn)
        sched_inn = torch.optim.lr_scheduler.ExponentialLR(optim_inn, gamma=self.params["optimization"]['gamma'])
        return [optim_inn], [sched_inn]

    def validation_step(self, rollouts, batch_idx):
        cond_frame = self.get_cond_frame(rollouts['images'])
        hgn_result, energy_per_step = self.get_sample_video(cond_frame, n_steps=20)
        video = hgn_result.reconstructed_rollout
        sampling_video = make_video_second_fix(video, self.params['optimization']['batch_size'],
                                               bair=self.params['dataset']['name'] == 'Bair')
        wandb.log({'val/sampling': wandb.Video(sampling_video, fps=1, format="gif")})
        self.log_dict({'val/energy_per_step': energy_per_step})

        self.features_fvd_fake_samples.append(video.detach().cpu().numpy())
        self.features_fvd_true_samples.append(
            rollouts['images'][:, :self.params['logging']['n_samples']].detach().cpu().numpy())

    def validation_epoch_end(self, outs):
        self.FVD.i3d.eval()

        features_fake_samples = torch.from_numpy(np.concatenate(self.features_fvd_fake_samples, axis=0))
        features_true_samples = torch.from_numpy(np.concatenate(self.features_fvd_true_samples, axis=0))
        fvd_score_samples = calculate_FVD(self.FVD.i3d, features_fake_samples, features_true_samples,
                                          batch_size=self.params["logging"]["bs_i3d"], cuda=True)

        self.features_fvd_fake_samples.clear()
        self.features_fvd_true_samples.clear()

        self.log_dict({'val/fvd_sampling': float(fvd_score_samples)})
        #np.random.seed(self.current_epoch)

    def set_model(self):
        pass

    def get_checkpoint_monitoring(self, logging_root):
        return ModelCheckpoint(dirpath=logging_root,
                               monitor='val/fvd_sampling',
                               mode='min',
                               auto_insert_metric_name=True)

