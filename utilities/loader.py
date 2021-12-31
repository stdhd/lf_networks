import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.hamiltonian_generative_network import HGN
from networks.decoder_net import DecoderNet
from networks.encoder_net import EncoderNet
#from networks.encoder_transformer import EncoderTransformerNet
from networks.hamiltonian_net import HamiltonianNet
from networks.transformer_net import TransformerNet
from utilities.integrator import Integrator


def instantiate_encoder(params, device, dtype):
    encoder = EncoderNet(seq_len=params["dataset"]["rollout"]["seq_length"],
                         in_channels=params["dataset"]["rollout"]["n_channels"],
                         **params["networks"]["encoder"],
                         dtype=dtype)#.to(device)
    return encoder


def instantiate_transformer(params, device, dtype):
    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"],
        dtype=dtype)#.to(device)
    return transformer


def instantiate_hamiltonian(params, device, dtype):
    hnn = HamiltonianNet(**params["networks"]["hamiltonian"],
                         dtype=dtype)#.to(device)
    return hnn


def instantiate_decoder(params, device, dtype):
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
        out_channels=params["dataset"]["rollout"]["n_channels"],
        **params["networks"]["decoder"],
        dtype=dtype)#.to(device)
    return decoder


def load_hgn(params, device, dtype):
    """Return the Hamiltonian Generative Network created from the given parameters.

    Args:
        params (dict): Experiment parameters (see experiment_params folder).
        device (str): String with the device to use. E.g. 'cuda:0', 'cpu'.
        dtype (torch.dtype): Data type to be used by the networks.
    """
    # Define networks
    if params["networks"]["encoder"]["type"] == 'transformer':
       # encoder = EncoderTransformerNet(seq_len=params["optimization"]["input_frames"],
       #                  in_channels=params["dataset"]["rollout"]["n_channels"],
       #                  **params["networks"]["encoder"],
       #                  dtype=dtype)
        pass
    else:


        encoder = EncoderNet(seq_len=params["optimization"]["input_frames"],
                             in_channels=params["dataset"]["rollout"]["n_channels"],
                             **params["networks"]["encoder"],
                             dtype=dtype)


    transformer = TransformerNet(
        in_channels=params["networks"]["encoder"]["out_channels"],
        **params["networks"]["transformer"],
        dtype=dtype)  # .to(device)


    hnn = HamiltonianNet(**params["networks"]["hamiltonian"],
                         dtype=dtype)
    decoder = DecoderNet(
        in_channels=params["networks"]["transformer"]["out_channels"],
        out_channels=params["dataset"]["rollout"]["n_channels"],
        **params["networks"]["decoder"],
        dtype=dtype)#.to(device)

    # Define HGN integrator
    integrator = Integrator(delta_t=params["dataset"]["rollout"]["delta_time"],
                            method=params["integrator"]["method"])
    
    # Instantiate Hamiltonian Generative Network
    hgn = HGN(encoder=encoder,
              transformer=transformer,
              hnn=hnn,
              decoder=decoder,
              integrator=integrator,
              device=device,
              dtype=dtype,
              seq_len=params["dataset"]["video_length"] - params['optimization']['input_frames'],
              channels=params["dataset"]["rollout"]["n_channels"],
              append_first_image=params["networks"]["append_first_image"])
    return hgn


