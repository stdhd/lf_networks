import wandb
from networks.hamiltonian_net import HamiltonianNet
from networks.hamiltonian_u_net import HamiltonianUNet
from networks.decoder_net import DecoderNet
from utilities.integrator import Integrator
from networks.conv_gru import ConvGRU
from networks.motion_encoder import MotionEncoder
from networks.startframe_encoder import get_encoder
from utilities.taming_utilities import *
from networks.openai_guided_diffusion.openai_unet import UNetModelNoEmbeddings
import random


class EndToEndPredictor(nn.Module):
    def __init__(self, params, *args, **kwargs,):
        super().__init__()
        self.params = params
        self.batch_size = params['optimization']['batch_size']
        if kwargs.get('load', False):
            pass

        else:
            self.start_encoder = get_encoder(in_channels=3, **self.params["networks"]['start_encoder'])
            self.motion_encoder = MotionEncoder(
                self.params["networks"]['motion_encoder'])

            if 'hamiltonian' in self.params['networks']:
                self.hnn = HamiltonianNet(**self.params['networks']['hamiltonian'])
            elif 'unet' in self.params['networks']:
                self.hnn = HamiltonianUNet(in_channels=2*params["latent_dimensions"]["pq_latent_size"],
                                     **self.params['networks']['unet'])
            elif 'oaiunet' in self.params['networks']:
                self.hnn = UNetModelNoEmbeddings(in_channels=2*params["latent_dimensions"]["pq_latent_size"],
                                           out_channels=2*params["latent_dimensions"]["pq_latent_size"],
                                     **self.params['networks']['oaiunet'])
            elif 'gru' in self.params['networks']:
                self.hnn = ConvGRU(**params['networks']['gru'])
            if params.get('watch_gradients', False):
                wandb.watch(self.hnn, log='all')

            if 'integrator' in params:
                self.integrator = Integrator(delta_t=params["dataset"]["rollout"]["delta_time"],
                                            method=params["integrator"]["method"],
                                            learn_delta_t=params["integrator"].get("learnable_delta_t", False),
                                             do_not_integrate_but_let_unet_sample=params["integrator"].get("do_not_integrate_but_let_unet_sample", False))
            self.decoder = DecoderNet(
                in_channels=params["latent_dimensions"]["pq_latent_size"],
                out_channels=params["dataset"]["rollout"]["n_channels"],
                **params['networks']['decoder'])
        self.use_quantization = False
        if 'codebook' in params:
            self.init_quantization()
            self.use_quantization = True


    def init_quantization(self):
        self.frame_quantize = VectorQuantizer2(self.params["codebook"]["n_embed"],
                                              self.params["codebook"]["embed_dim"], beta=0.25)
        self.quant_conv = torch.nn.Conv2d(32, self.params["codebook"]["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(self.params["codebook"]["embed_dim"], 32, 1)

    def quantize_p(self, h, prediction=None):
        if self.use_quantization:
            quant = self.quant_conv(h)
            quant, emb_loss, (perplexity, min_encodings, min_encoding_indices) = self.frame_quantize(quant)
            quant = self.post_quant_conv(quant)
            usages = torch.unique(min_encoding_indices)
            prediction.set_codebook_usages(usages)
            prediction.add_emb_loss(emb_loss)
            return quant
        else:
            return h

    def forward(self, rollout_batch=None, conditioning_frame=None, prediction=None, n_steps=4, initial_p=None, p_use_true_q=-1):
        if initial_p is None:
            motion, mu, logvar = self.motion_encoder(rollout_batch[:, :])
            p = motion.view((-1, self.params['latent_dimensions']['pq_latent_size'],
                                self.params['latent_dimensions']['pq_latent_dim'],
                                self.params['latent_dimensions']['pq_latent_dim']))
            p = self.quantize_p(p, prediction=prediction)
            prediction_shape = list(rollout_batch.shape)
        else:
            p = self.quantize_p(initial_p, prediction=prediction)
            mu = None
            logvar = None
            prediction_shape = list(conditioning_frame.shape)
        prediction_shape[1] = n_steps
        prediction.set_z(z_sample=p, z_mean=mu, z_logvar=logvar)
        previously_predicted_frame = conditioning_frame
        previous_true_frame = conditioning_frame
        energy_sum = 0
        hidden_state = None
        for i in range(n_steps):
            if i > 0 and self.params['optimization'].get('use_integrated_q_instead_of_encoded', False):
                q_estimated = q
            else:
                q_estimated = self.start_encoder(previously_predicted_frame) # TODO: WHY NOT USE PREVIOUSLY PREDICTED q?
            if rollout_batch is not None:
                prediction.append_q_encoding(q_estimated, self.start_encoder(previous_true_frame))
                previous_true_frame = rollout_batch[:, i]
            if i > 0 and rollout_batch is not None:
                q_true = self.start_encoder(rollout_batch[:, i - 1])
            if random.uniform(0., 1.) <= p_use_true_q and i > 0:
                q = q_true
                if self.params['optimization'].get('spade_frame_equals_previous', False):
                    spade_frame = rollout_batch[:, i - 1]
                else:
                    spade_frame = conditioning_frame
            else:
                q = q_estimated
                if self.params['optimization'].get('spade_frame_equals_previous', False):
                    spade_frame = previously_predicted_frame
                else:
                    spade_frame = conditioning_frame

            if (self.params['optimization'].get('subsample_rollouts', False)):
                for j in range(9):
                    q, p, energy, hidden_state = self.integrate_step(q, p, i, hidden_state=hidden_state)
            q, p, energy, hidden_state = self.integrate_step(q, p, i, hidden_state=hidden_state)
            p = self.quantize_p(p, prediction=prediction)
            x_reconstructed = self.decoder(q, spade_frame)
            prediction.append_reconstruction(x_reconstructed)
            prediction.append_energy(energy)
            prediction.append_state(q, p)

            if energy is not None:
                energy_sum += prediction.get_energy()[0]
            if self.params['optimization'].get('detach_previous_frame_and_q', False):
                previously_predicted_frame = x_reconstructed.detach()
            else:
                previously_predicted_frame = x_reconstructed

        return prediction, energy_sum / n_steps

    def integrate_step(self, q, p, step_no, hidden_state):
        energy = None
        hidden_state = None
        if 'hamiltonian' in self.params['networks']:
            q, p, energy = self.integrator._lf_step(q=q, p=p, hnn=self.hnn)
        else:
            q, p = self.integrator.step_non_scalar(q=q, p=p, hnn=self.hnn)

        return q, p, energy, hidden_state

    def post_process_rollouts_to_q(self, q, p):
        return q

    def sample(self, p, conditioning_frame, n_steps, prediction):
        previous_frame = conditioning_frame
        q = self.start_encoder(conditioning_frame)
        hidden_state = None
        for i in range(n_steps):
            q, p, energy, hidden_state = self.integrate_step(q, p, i, hidden_state=hidden_state)
            x_reconstructed = self.decoder(self.post_process_rollouts_to_q(q, p),
                                           previous_frame)
            prediction.append_reconstruction(x_reconstructed)
            if self.params['optimization'].get('spade_frame_equals_previous', False):
                previous_frame = x_reconstructed
        return prediction