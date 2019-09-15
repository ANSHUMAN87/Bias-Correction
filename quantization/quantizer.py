import torch
from torch.autograd import Function
import numpy as np
import math

def to_cuda(t, device):
    if isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        return torch.tensor(t, dtype=torch.float32).to(device)

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor

def QuantizeWeightsPerChannel(tensor, min_=None, max_=None):
    # Assume weights with dimensions [OFM,IFM,K1,K2]
    t = tensor.view(tensor.shape[0], -1)

    # per output channel min, max
    if min_ is None:
        min_ = t.min(-1)[0]
    if max_ is None:
        max_ = t.max(-1)[0]

    output = Quantize(t, max_ - min_, min_)

    return output.view(tensor.shape)

def QuantizeActivationPerChannel(tensor, tag="", stat_id=None, min_=None, max_=None):
    if min_ is None:
        min_ = Minmax_perchannel(tensor, ['min'])['min']
    min_ = to_cuda(min_, tensor.device)

    if max_ is None:
        max_ = Minmax_perchannel(tensor, ['max'])['max']
    max_ = to_cuda(max_, tensor.device)

    N, C, H, W = tensor.shape  # N x C x H x W
    t = tensor.detach().transpose(0, 1).contiguous()  # C x N x H x W
    t = t.view(t.shape[0], -1)

    output = Quantize(t, max_ - min_, min_)
    output = output.view(C, N, H, W).transpose(0, 1).contiguous()  # N x C x H x W
    return output.view(tensor.shape)
    
def Minmax_perchannel(tensor, stats):
    # Assume activation dimensions [N,C,H,W]
    t = tensor.transpose(0, 1).contiguous()  # [C, N, H, W]
    t = t.view(t.shape[0], -1) # [C, NxHxW]

    stats_dict = {}
    for s in stats:
        if s == 'max':
            stats_dict[s] = t.max(dim=-1)[0]
        elif s == 'min':
            stats_dict[s] = t.min(dim=-1)[0]
        elif s == 'mean':
            stats_dict[s] = t.mean(dim=-1)
        elif s == 'b':
            stats_dict[s] = torch.mean(torch.abs(t - t.mean(dim=-1).unsqueeze(-1)), dim=-1)
        elif s == 'std':
            stats_dict[s] = torch.std(t, dim=-1, unbiased=True)

    return stats_dict

def Quantize(tensor, delta, offset):
    qmin = 0.
    qmax = 2.**8 - 1.

    scale = (delta) / (qmax - qmin)

    scale = torch.max(scale, torch.tensor([1e-8]).to(scale.device))

    output = tensor.detach()
    
    initial_zero_point = qmin - offset / scale
    zero_point = torch.round(initial_zero_point)

    output = torch.div(output, scale.unsqueeze(-1))
    output = torch.add(output, zero_point.unsqueeze(-1))
    output.clamp_(qmin, qmax).round_()  # quantize

    output = torch.add(output, -zero_point.unsqueeze(-1))
    output = torch.mul(output, scale.unsqueeze(-1))  # dequantize

    return output.view(tensor.shape)
