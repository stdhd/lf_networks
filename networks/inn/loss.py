# taken from https://github.com/ablattmann/inn_poking/blob/366bd8b706cae99bb42ea676391da58797657bc5/models/modules/INN/loss.py#L6

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as f


class FlowLoss(nn.Module):
    def __init__(self,spatial_mean=False, logdet_weight=1.):
        super().__init__()
        # self.config = config
        self.spatial_mean = spatial_mean
        self.logdet_weight = logdet_weight

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample, spatial_mean=self.spatial_mean))
        assert len(logdet.shape) == 1
        if self.spatial_mean:
            h,w = sample.shape[-2:]
            nlogdet_loss = -torch.mean(logdet) / (h*w)
        else:
            nlogdet_loss = -torch.mean(logdet)

        loss = nll_loss + self.logdet_weight*nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample),spatial_mean=self.spatial_mean))

        kld = f.kl_div(sample, torch.randn_like(sample), reduce=True)
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss": nll_loss,
            'logdet_weight': self.logdet_weight,
            'kld_match_gaussian': kld
        }
        return loss, log


def nll(sample, spatial_mean= False):
    if spatial_mean:
        return 0.5 * torch.sum(torch.mean(torch.pow(sample, 2),dim=[2,3]), dim=1)
    else:
        return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])