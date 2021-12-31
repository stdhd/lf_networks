"""Script to train the Hamiltonian Generative Network
"""
import os
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pytorch_lightning as pl
from data.moving_mnist.moving_mnist import MovingMNIST
from data.bair.bair import BairDataset
from torchvision import transforms

from utilities.loader import load_hgn
from losses import reconstruction_loss, kld_loss, geco_constraint, \
    geco_constraint_vggloss
from utilities.statistics import mean_confidence_interval
import utilities.loader as loader

from logger.custom_logging import make_video
from losses import VGGLoss


class HgnTrainer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        #self.inception_model = init_model()

        self.params = kwargs.pop('params')
        self.resume = kwargs.pop('resume')
        self.batch_size = self.params['optimization']['batch_size']
        dataset_name = self.params['dataset']['dataset_name']
        if dataset_name == 'bair':
            self.train_ds = BairDataset(self.params, mode='train')
            self.test_ds = BairDataset(self.params, mode='test')
        else:
            self.train_ds = MovingMNIST(self.params, train=True, download=True, transform=transforms.ToTensor())
            self.test_ds = MovingMNIST(self.params, train=False, download=False, transform=transforms.ToTensor())

        if not self.resume:  # Fail if experiment_id already exist in runs/
            self._avoid_overwriting(self.params["experiment_id"])
            wandb.run.name = self.params["experiment_id"]


        # Get dtype, will raise a 'module 'torch' has no attribute' if there is a typo
        #self.dtype = torch.to(params["networks"]["dtype"]) TODO
        # Load hgn from parameters to deice
        self.hgn = load_hgn(params=self.params,
                            device=self.device,
                            dtype=self.dtype)

        self.vggloss = VGGLoss()

        if 'load_path' in self.params:
            self.load_and_reset(self.params, self.device, self.dtype)

        if self.params["networks"]["variational"]:
            self.langrange_multiplier = self.params["geco"]["initial_lagrange_multiplier"]
            self.C_ma = None

    def configure_optimizers(self):
        # Define optimization modules
        params = self.params
        optim_params = [
            {
                'params': self.hgn.encoder.parameters(),
                'lr': params["optimization"]["encoder_lr"]
            },
            {
                'params': self.hgn.transformer.parameters(),
                'lr': params["optimization"]["transformer_lr"]
            },
            {
                'params': self.hgn.hnn.parameters(),
                'lr': params["optimization"]["hnn_lr"]
            },
            {
                'params': self.hgn.decoder.parameters(),
                'lr': params["optimization"]["decoder_lr"]
            },
        ]
        optimizer = torch.optim.Adam(optim_params)
        return optimizer

    def load_and_reset(self, params, device, dtype):
        """Load the HGN from the path specified in params['load_path'] and reset the networks in
        params['reset'].

        Args:
            params (dict): Dictionary with all the necessary parameters to load the networks.
            device (str): 'gpu:N' or 'cpu'
            dtype (torch.dtype): Data type to be used in computations.
        """
        self.hgn.load(params['load_path'])
        if 'reset' in params:
            if isinstance(params['reset'], list):
                for net in params['reset']:
                    assert net in ['encoder', 'decoder', 'hamiltonian', 'transformer']
            else:
                assert params['reset'] in ['encoder', 'decoder', 'hamiltonian', 'transformer']
            if 'encoder' in params['reset']:
                self.hgn.encoder = loader.instantiate_encoder(params, device, dtype)
            if 'decoder' in params['reset']:
                self.hgn.decoder = loader.instantiate_decoder(params, device, dtype)
            if 'transformer' in params['reset']:
                self.hgn.transformer = loader.instantiate_transformer(params, device, dtype)
            if 'hamiltonian' in params['reset']:
                self.hgn.hnn = loader.instantiate_hamiltonian(params, device, dtype)

    def training_step(self, rollouts, batch_idx):
        """Perform a training step with the given rollouts batch.

        Args:
            rollouts (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                corresponding to a batch of sampled rollouts.

        Returns:
            A dictionary of losses and the model's prediction of the rollout. The reconstruction loss and
            KL divergence are floats and prediction is the HGNResult object with data of the forward pass.
        """
        rollouts = rollouts['seq']
        rollout_len = rollouts.shape[1]
        input_frames = self.params['optimization']['input_frames']

        assert (
                    input_frames <= rollout_len)  # optimization.use_steps must be smaller (or equal) to rollout.sequence_length
        roll = rollouts[:, :input_frames]

        hgn_output = self.hgn.forward(rollout_batch=roll, n_steps=rollout_len - input_frames)
        target = rollouts[:,
                 input_frames - 1:]  # Fit first input_frames and try to predict the last + the next (rollout_len - input_frames)

        prediction = hgn_output.reconstructed_rollout

        if self.params["networks"]["variational"]:
            tol = self.params["geco"]["tol"]
            alpha = self.params["geco"]["alpha"]
            lagrange_mult_param = self.params["geco"]["lagrange_multiplier_param"]
            if self.params["loss"] == "perceptual":
                C, rec_loss = geco_constraint_vggloss(target, prediction, tol, self.vggloss)  # C has gradient
            else:
                C, rec_loss = geco_constraint(target, prediction, tol)  # C has gradient

            # Compute moving average of constraint C (without gradient)
            if self.C_ma is None:
                self.C_ma = C.detach()
            else:
                self.C_ma = alpha * self.C_ma + (1 - alpha) * C.detach()
            C_curr = C.detach().item()  # keep track for logging
            C = C + (self.C_ma - C.detach())  # Move C without affecting its gradient

            # Compute KL divergence
            mu = hgn_output.z_mean
            logvar = hgn_output.z_logvar
            kld = kld_loss(mu=mu, logvar=logvar)

            # normalize by number of frames, channels and pixels per frame
            kld_normalizer = prediction.flatten(1).size(1)
            kld = kld / kld_normalizer

            # Compute losses
            train_loss = kld + self.langrange_multiplier * C

            # clamping the langrange multiplier to avoid inf values
            self.langrange_multiplier = self.langrange_multiplier * torch.exp(
                lagrange_mult_param * C.detach())
            self.langrange_multiplier = torch.clamp(self.langrange_multiplier, 1e-10, 1e10)

            losses = {
                'train/loss': train_loss.item(),
                'train/kld': kld.item(),
                'train/C': C_curr,
                'train/C_ma': self.C_ma.item(),
                'train/rec': rec_loss.item(),
                'train/langrange_mult': self.langrange_multiplier.item()
            }


        else:
            # not variational
            # Compute frame reconstruction error
            if self.params["loss"] == "perceptual":
                combined_shape = prediction.shape[0] * prediction.shape[1], prediction.shape[2], prediction.shape[3], prediction.shape[4]
                train_loss = 0
                for i in range(prediction.shape[0]):
                    train_loss += self.vggloss(prediction[i], target[i])
                train_loss = train_loss / prediction.shape[0]
            else:
                train_loss = reconstruction_loss(
                    target=target,
                    prediction=prediction)
            losses = {'loss': train_loss.item()}
        wandb.log(losses)
        # make_dot(train_loss).render("attached", format="png")
        return train_loss

    def compute_reconst_kld_errors_per_batch(self, rollout_batch):
        """Computes reconstruction error and KL divergence.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader to retrieve errors from.

        Returns:
            (reconst_error_mean, reconst_error_h), (kld_mean, kld_h): Tuples where the mean and 95%
            conficence interval is shown.
        """
        first = True

        # Pytorch lightning disables gradient tracing during validation - but we need the gradient in the model
        torch.set_grad_enabled(True)
        # Move to device and change dtype
        rollout_batch = rollout_batch['seq']
        rollout_batch = Variable(rollout_batch, requires_grad=True)  # .to(self.device).type(self.dtype)
        rollout_len = rollout_batch.shape[1]
        input_frames = self.params['optimization']['input_frames']
        assert (
                input_frames <= rollout_len)  #   optimization.use_steps must be smaller (or equawandb.logl) to rollout.sequence_length


        roll = rollout_batch[:, :input_frames]
        hgn_output = self.hgn.forward(rollout_batch=roll, n_steps=rollout_len - input_frames)
        # TODO: Number of rollouts yields n_steps+1
        target = rollout_batch[:,
                 input_frames - 1:]#.cuda()  # Fit first input_frames and try to predict the last + the next (rollout_len - input_frames)

        prediction = hgn_output.reconstructed_rollout.clone().detach()

        error = reconstruction_loss(
            target=target,
            prediction=prediction, mean_reduction=False).detach().cpu(
        ).clone().detach().numpy()
        set_errors = None
        set_klds = None
        wandb.log({"val/reconstruction_loss": error})

        if self.params["networks"]["variational"]:
            kld = kld_loss(mu=hgn_output.z_mean, logvar=hgn_output.z_logvar, mean_reduction=False).detach().cpu(
            ).numpy()
            # normalize by number of frames, channels and pixels per frame
            kld_normalizer = prediction.flatten(1).size(1)
            kld = kld / kld_normalizer
        if first:
            first = False
            set_errors = error
            if self.params["networks"]["variational"]:
                set_klds = kld
        else:
            set_errors = np.concatenate((set_errors, error))
            if self.params["networks"]["variational"]:
                set_klds = np.concatenate((set_klds, kld))
        err_mean, err_h = mean_confidence_interval(set_errors)
        err_dict = {}
        if self.params["networks"]["variational"]:
            kld_mean, kld_h = mean_confidence_interval(set_klds)
            err_dict = {
             'val/err_mean': err_mean,
             'val/err_h': err_h,
             'val/kld_kld_mean': kld_mean,
             'val/kld_h': kld_h}

        else:
            err_dict = {
                'val/err_mean': err_mean,
                'val/err_h': err_h,}

        return err_dict, prediction

    def validation_step(self, rollouts, batch_idx):

        err_dict, prediction = self.compute_reconst_kld_errors_per_batch(rollouts)
        wandb.log(err_dict)
        # Save first sequence prediction
        input_frames = self.params['optimization']['input_frames']
        return prediction, rollouts

    def validation_epoch_end(self, outs):
        n_videos_logged = 1
        predicted = outs[0][0]
        input_frames = self.params['optimization']['input_frames']
        target = outs[0][1]['seq'][:, input_frames-1:]

        rollout_len = target.size(1) - input_frames

        if self.params['logging']['video']:
            validation_video = make_video(target[:n_videos_logged, :], predicted[:n_videos_logged, :],
                                          n_videos_logged)
            wandb.log({"video": wandb.Video(validation_video, fps=1, format="gif")})

    def test(self):
        """Test after the training is finished and logs result to tensorboard.
        """
        print("Calculating final training error...")
        (err_mean, err_h), kld = self.compute_reconst_kld_errors(self.train_data_loader)
        if kld is not None:
            kld_mean, kld_h = kld
            #self.training_logger.log_error("Train KL divergence", kld_mean, kld_h)

        print("Calculating final test error...")
        (err_mean, err_h), kld = self.compute_reconst_kld_errors(self.test_data_loader)
        #self.training_logger.log_error("Test reconstruction error", err_mean, err_h)
        if kld is not None:
            kld_mean, kld_h = kld
            #self.training_logger.log_error("Test KL divergence", kld_mean, kld_h)

    def train_data_loader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.params['num_workers'])

    def validation_data_loader(self):
        return DataLoader(self.test_ds, batch_size=self.params['optimization']['batch_size'], num_workers=3)

    def _avoid_overwriting(self, experiment_id):
        # This function throws an error if the given experiment data already exists in runs/

        last_number = 1
        while True:
            if not os.path.exists(os.path.join('../runs', f'{experiment_id}_{last_number}')):
                break
            else:
                last_number += 1
        self.params["experiment_id"] = f'{experiment_id}_{last_number}'


