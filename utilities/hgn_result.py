from utilities import conversions
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HgnResult():
    """Class to bundle HGN guessed output information.
    """

    def __init__(self, batch_shape, device, q_shape=(32, 8, 8), n_codes=2048, log_states=False):
        """Instantiate the HgnResult that will contain all the information of the forward pass
        over a single batch of rollouts.

        Args:
            batch_shape (torch.Size): Shape of a batch of reconstructed rollouts, returned by
                batch.shape.
            device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
        """
        self.input = None
        self.z_mean = None
        self.z_logvar = None
        self.z_sample = None
        self.q_s = []
        self.p_s = []
        self.energies = []  # Estimated energy of the system by the Hamiltonian network
        self.reconstructed_rollout = torch.empty(batch_shape, device=device)
        single_frame_shape = batch_shape
        self.start_frame = torch.empty(batch_shape[2:], device=device)
        self.q_estimated = torch.empty((batch_shape[0], batch_shape[1], *q_shape), device=device)
        self.q_true = torch.empty((batch_shape[0], batch_shape[1], *q_shape), device=device)
        self.log_states = log_states
        if log_states:
            self.q = torch.empty((batch_shape[0], batch_shape[1], *q_shape), device=device)
            self.p = torch.empty((batch_shape[0], batch_shape[1], *q_shape), device=device)
        #self.hist_codebook_usages = torch.zeros(n_codes, device="cpu", dtype=torch.int64)
        self.codebook_usages = []
        self.n_codes = n_codes
        self.reconstruction_ptr = 0
        self.emb_loss = torch.empty(1, device=device)
        self.q_ptr = 0
        self.state_ptr = 0

    def clear(self):
        self.input = None
        self.z_mean = None
        self.z_logvar = None
        self.z_sample = None
        self.q_s = []
        self.p_s = []
        self.energies = []  # Estimated energy of the system by the Hamiltonian network
        self.reconstruction_ptr = 0
        self.q_ptr = 0

    def set_start_frame (self, start):
        self.start_frame = start

    def set_input(self, rollout):
        """Store ground truth of system evolution.

        Args:
            rollout (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                containing the ground truth rollouts of a batch.
        """
        self.input = rollout

    def set_z(self, z_sample, z_mean=None, z_logvar=None):
        """Store latent encodings and correspondent distribution parameters.

        Args:
            z_sample (torch.Tensor): Batch of latent encodings.
            z_mean (torch.Tensor, optional): Batch of mens of the latent distribution.
            z_logvar (torch.Tensor, optional): Batch of log variances of the latent distributions.
        """
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.z_sample = z_sample

    def set_emb_loss(self, emb_loss):
        self.emb_loss = emb_loss

    def add_emb_loss(self, emb_loss):
        if not emb_loss is None:
            self.emb_loss += emb_loss

    def append_state(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list .

        Args:
            q (torch.Tensor): Tensor with the abstract position.
            p (torch.Tensor): Tensor with the abstract momentum.
        """
        if self.log_states:
            self.p[:, self.state_ptr] = p
            self.q[:, self.state_ptr] = q
            self.state_ptr += 1
            #self.q_s.append(q)
            #self.p_s.append(p)#

    def set_codebook_usages(self, usages):
        #add_hist = torch.histc(usages.detach().to(torch.int64), bins=self.n_codes, min=0, max=self.n_codes).cpu()
        self.codebook_usages = self.codebook_usages + (usages.detach().cpu().numpy().tolist())

    def get_unique_codebook_ids(self):
        return len(set(self.codebook_usages))

    def append_reconstruction(self, reconstruction):
        """Append guessed reconstruction.

        Args:
            reconstruction (torch.Tensor): Tensor of shape (seq_len, channels, height, width).
                containing the reconstructed rollout.
        """
        assert self.reconstruction_ptr < self.reconstructed_rollout.shape[1],\
            'Trying to add rollout number ' + str(self.reconstruction_ptr) + ' when batch has ' +\
            str(self.reconstructed_rollout.shape[1])
        self.reconstructed_rollout[:, self.reconstruction_ptr] = reconstruction
        self.reconstruction_ptr += 1

    def append_q_encoding(self, q_estimated, q_true):
        assert self.q_ptr < self.reconstructed_rollout.shape[1],\
            'Trying to add rollout number ' + str(self.q_ptr) + ' when batch has ' +\
            str(self.reconstructed_rollout.shape[1])
        self.q_estimated[:, self.q_ptr] = q_estimated
        self.q_true[:, self.q_ptr] = q_true
        self.q_ptr += 1

    def append_energy(self, energy):
        """Append the guessed system energy to energy list.

        Args:
            energy (torch.Tensor): Energy of each trajectory in the batch.
        """
        self.energies.append(energy)

    def get_energy(self):
        """Get the average energy of that rollout and the average of each trajectory std.

        Returns:
            (tuple(float, float)): (average_energy, average_std_energy) average_std_energy is computed as follows:
            For each trajectory in the rollout, compute the std of the energy and average across trajectories.
        """
        energies = np.array(self.energies) 
        energy_std = np.std(energies, axis=0)
        return np.mean(energies), np.mean(energy_std)

