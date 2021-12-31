import torch
import math
from losses import reconstruction_loss, kld_loss, RecLoss
from networks.e2e_predictor import EndToEndPredictor
from networks.e2e_crossattention_predictor import CrossAttPredictor
from experiments.BaseExperiment import BaseExperiment
from utilities.hgn_result import HgnResult


class FirstStageTrainer(BaseExperiment):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.sample_prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                                   self.params['logging']['n_samples'], 3, 64, 64)),
                                           device=self.device)

        self.prediction = HgnResult(batch_shape=torch.Size((self.batch_size,
                                                            self.n_input_frames, 3, 64, 64)),
                                    device=self.device)
        self.automatic_optimization = True

        self.loss = RecLoss(**params['loss_weights'])

    def set_model(self):
        predictors = {'EndToEnd': EndToEndPredictor, 'CA': CrossAttPredictor}
        self.model = predictors[self.params.get('predictor', 'EndToEnd')](self.params)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        rollouts = batch['images']
        rollout_len = rollouts.shape[1]

        assert (self.n_input_frames <= rollout_len)

        input = self.get_input_frames(rollouts)
        target = self.get_target_frames(rollouts)
        cond_frame = self.get_cond_frame(rollouts)
        n_steps = target.size(1)

        if self.params['optimization'].get('scheduled_sampling', False):
            p_use_true_q = -1 / (1 + math.exp(-0.5 * self.current_epoch + 5)) + 1  #min(1., 3/max(self.current_epoch, 0.1))
        else:
            p_use_true_q = self.params['optimization'].get('prob_use_true_q', 0.)
        self.log('train/p_use_true_q', p_use_true_q)

        self.prediction = HgnResult(batch_shape=torch.Size((input.size(0),
                                                            self.n_input_frames, 3, 64, 64)),
                                    device=self.device)

        hgn_output, energy_mean = self.model.forward(rollout_batch=input, conditioning_frame=cond_frame,
                                                     prediction=self.prediction,
                                                     n_steps=n_steps, p_use_true_q=p_use_true_q)

        if 'codebook' in self.params:
            self.log_dict({'train/used_codebook_p': hgn_output.get_unique_codebook_ids(),
                           'train/used_codebook_ids': hgn_output.codebook_usages})

        if 'integrator' in self.params:
            self.log_dict({'train/mean_energies': energy_mean,
                           'train/delta_t': self.model.integrator.delta_t})
        prediction = hgn_output.reconstructed_rollout

        if optimizer_idx == 0:
            gen_loss, log_dict_gen = self.loss(hgn_output.emb_loss, target,
                                               prediction,
                                               optimizer_idx, self.global_step,
                                               last_layer=self.get_last_layer(), split="train",
                                               q_encoding_estimated=hgn_output.q_estimated,
                                               q_encoding_true=hgn_output.q_true)
            self.log('train/gen_loss', gen_loss)
            self.log_dict(log_dict_gen)
            return gen_loss

        else:
            disc_loss, log_dict_disc = self.loss(hgn_output.emb_loss, target,
                                               prediction,
                                               optimizer_idx, self.global_step,
                                               last_layer=self.get_last_layer(), split="train",
                                               q_encoding_estimated=hgn_output.q_estimated,
                                               q_encoding_true=hgn_output.q_true
                                                 )
            self.log_dict(log_dict_disc)
            self.log('train/disc_loss', disc_loss, on_step=True)
            return disc_loss

    def get_last_layer(self):
        return list(self.model.decoder.parameters())[-1]

    def configure_optimizers(self):
        params = self.params
        params_g = [
            {
                'params': self.model.start_encoder.parameters(),
                'lr': params["optimization"]["frame_encoder_lr"],
                'name': 'start frame encoder',
                #'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.motion_encoder.parameters(),
                'lr': params["optimization"]["motion_encoder_lr"],
                'name': 'motion encoder',
                #'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.hnn.parameters(),
                'lr': params["optimization"]["hnn_lr"],
                'name': 'HNN or UNET',
                #'weight_decay': params["optimization"].get("weight_decay", 0),
            },
            {
                'params': self.model.decoder.parameters(),
                'lr': params["optimization"]["decoder_lr"],
                #'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'decoder'
            },
            {
                'params': self.loss.logvar,
                'lr': params["optimization"]["decoder_lr"],
                # 'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'logvar'
            },
        ]

        if 'integrator' in params:
            params_g.append({
                'params': self.model.integrator.parameters(),
                'lr': params["optimization"]["integrator_lr"],
                #'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'integrator'
            },)

        if 'context_encoder' in params['networks']:
            params_g.append({
                'params': self.model.state_to_context_encoder.parameters(),
                'lr': params["optimization"]["hnn_lr"],
                #'weight_decay': params["optimization"].get("weight_decay", 0),
                'name': 'state_to_context_encoder'
            }

            )

        if 'codebook' in params:
            params_g.append(
                {
                    'params': self.model.frame_quantize.parameters(),
                    'lr': params["optimization"]["codebook_lr"],
                    #'weight_decay': params["optimization"].get("weight_decay", 0),
                    'name': 'quantizer'
                },

            )

            params_g.append(
                {
                    'params': self.model.quant_conv.parameters(),
                    'lr': params["optimization"]["codebook_lr"],
                    #'weight_decay': params["optimization"].get("weight_decay", 0),
                    'name': 'quant_conv'
                },

            )

            params_g.append(
                {
                    'params': self.model.post_quant_conv.parameters(),
                    'lr': params["optimization"]["codebook_lr"],
                    #'weight_decay': params["optimization"].get("weight_decay", 0),
                    'name': 'post_quant_conv'
                },

            )

        params_disc = [
            {
                'params': self.loss.discriminator.parameters(),
                'lr': params["optimization"]["disc_t_lr"],
                'name': 'discriminator'
            }
        ] if self.params['loss_weights'].get('enable_adversarial', False) else []

        optim_g = torch.optim.Adam(params_g)
        if self.params['loss_weights'].get('enable_adversarial', False):
            optim_disc = torch.optim.Adam(params_disc)  # TODO: Add weight decay value

        sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.params["optimization"]['gamma'])
        if self.params['loss_weights'].get('enable_adversarial', False):
            sched_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=self.params["optimization"]["gamma"])

        if self.params['optimization'].get('use_scheduler', True):
            return [optim_g, optim_disc] if self.params['loss_weights'].get('enable_adversarial', False) else [optim_g], \
                   [sched_g, sched_disc] if self.params['loss_weights'].get('enable_adversarial', False) else [sched_g]
        else:
            return [optim_g, optim_disc] if self.params['loss_weights'].get('enable_adversarial', False) else [optim_g], \
                   []


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







