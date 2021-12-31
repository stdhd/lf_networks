import torch.nn as nn, torch
from networks.inn.blocks import *
from networks.inn.macow2 import MaCowStep,MultiScaleInternal, MultiscaleStack, MultiscaleMixCDF,HierarchicalConvCouplingFlow

class UnconditionalMaCowFlow(nn.Module):

    def __init__(self,config):
        super().__init__()
        channels = config["flow_in_channels"]
        hidden_dim = config["flow_mid_channels"]
        num_blocks = config["flow_hidden_depth"]
        coupling_type = config["coupling_type"]
        kernel_size = config["kernel_size"]
        heads = config["flow_attn_heads"]
        scale= config["scale"]
        self.n_flows = config["n_flows"]
        self.reshape = "none"
        self.sub_layers = nn.ModuleList()

        for i, flow in enumerate(range(self.n_flows)):

            self.sub_layers.append(UnconditionalMaCowFLowBlock(channels=channels,kernel_size=kernel_size,hidden_channels=hidden_dim,
                                                               scale=scale,heads=heads,coupling_type=coupling_type,
                                                               num_blocks=num_blocks))

    def forward(self, x, reverse=False):
        x = x.reshape((x.size(0), 8, 16, 16))
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
            self.last_logdets.append(logdet)
            x = x.reshape((x.size(0), 32, 8, 8))

            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
                if isinstance(x, tuple):
                    x = x[0]

            x = x.reshape((x.size(0), 32, 8, 8))
            return x

    def reverse(self, out):
        return self(out, reverse=True)

class UnsupervisedConvTransformer(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config
        self.flow = UnconditionalMixCDFConvFlow(self.config)

    def forward(self,input,reverse=False):
        if reverse:
            return self.reverse(input)
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample

class UnconditionalFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, activation='lrelu'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, activation=activation)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class ConditionalFlow(nn.Module):
    """Flat"""
    def __init__(self, in_channels, hidden_dim, hidden_depth, n_flows, activation='lrelu', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.sub_layers = nn.ModuleList()

        for flow in range(self.n_flows):
            self.sub_layers.append(UnconditionalFlatDoubleCouplingFlowBlock(
                                   self.in_channels, self.mid_channels,
                                   self.num_blocks, activation=activation)
                                   )

    def forward(self, x, reverse=False):
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, reverse=True)
            return x

    def reverse(self, out):
        return self(out, reverse=True)


class ConditionalConvFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""
    def __init__(self, in_channels=32, embedding_dim=32, hidden_dim=8, hidden_depth=8,
                 n_flows=6, conditioning_option="none", activation='lrelu', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = embedding_dim
        self.mid_channels = hidden_dim
        self.num_blocks = hidden_depth
        self.n_flows = n_flows
        self.conditioning_option = conditioning_option
        self.reshape = 'none'

        self.sub_layers = nn.ModuleList()
        for flow in range(self.n_flows):
            self.sub_layers.append(ConditionalConvDoubleCouplingFlowBlock(
                                   self.in_channels, self.cond_channels, self.mid_channels,
                                   self.num_blocks, activation=activation))

    def forward(self, x, embedding, reverse=False):
        hcond = embedding
        self.last_outs = []
        self.last_logdets = []
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x, logdet_ = self.sub_layers[i](x, hcond)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                if len(x.shape) != 4:
                    x = x.unsqueeze(-1).unsqueeze(-1)
                x = self.sub_layers[i](x, hcond, reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class Nvp(nn.Module):
    def __init__(self, blocks=[]):
        super(Nvp, self).__init__()

    def forward(self, x, condition):
        pass

    def get_det(self):
        pass


class UnsupervisedMaCowTransformer3(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.flow = MultiScaleInternal(MaCowStep,num_steps=self.config["num_steps"],in_channels=self.config["flow_in_channels"],
                                       hidden_channels=self.config["flow_mid_channels"],h_channels=0,
                                       factor=self.config["factor"],transform=self.config["transform"],
                                       prior_transform=self.config["prior_transform"],kernel_size=self.config["kernel_size"],
                                       coupling_type=self.config["coupling_type"],activation=self.config["activation"],
                                       use_1x1=self.config["use1x1"] if "use1x1" in self.config else False)

    def forward(self,input,reverse=False):
        if reverse:
            result = self.reverse(input)
            return result
        out, logdet = self.flow(input)
        return out, logdet

    def reverse(self,out):
        return self.flow(out,reverse=True)

    def sample(self,shape,device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde)
        return sample


class SupervisedMacowTransformer(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        condition_nice = 'condition_nice' in self.config and self.config['condition_nice']
        attention = 'attention' in self.config and self.config['attention']
        heads = self.config['flow_attn_heads']
        ssize = 'ssize' in self.config and self.config['ssize']
        cond_conv = 'cond_conv' in config and config['cond_conv']
        cond_conv_hidden_channels = config['cond_conv_hidden_channels'] if 'cond_conv_hidden_channels' else None
        p_drop = config['p_dropout'] if 'p_dropout' in config else 0.
        self.flow = MultiScaleInternal(MaCowStep, num_steps=self.config["num_steps"], in_channels=self.config["flow_in_channels"],
                                       hidden_channels=self.config["flow_mid_channels"], h_channels=self.config["h_channels"],
                                       factor=self.config["factor"], transform=self.config["transform"],
                                       prior_transform=self.config["prior_transform"], kernel_size=self.config["kernel_size"],
                                       coupling_type=self.config["coupling_type"], activation=self.config["activation"],
                                       use_1x1=self.config["use1x1"] if "use1x1" in self.config else False,
                                       condition_nice=condition_nice,attention=attention,heads=heads,spatial_size=ssize,
                                       cond_conv=cond_conv,cond_conv_hidden_channels=cond_conv_hidden_channels,
                                       p_dropout=p_drop
                                       )

    def forward(self, input, cond, reverse=False):
        if reverse:
            return self.reverse(input,cond)
        out, logdet = self.flow(input,cond)
        return out, logdet

    def reverse(self, out, cond):
        return self.flow(out, cond, reverse=True)

    def sample(self, shape,cond, device="cpu"):
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde,cond)
        return sample

