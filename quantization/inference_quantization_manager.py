import torch
import torch.nn as nn
from quantization.quantizer import QuantizeWeightsPerChannel, QuantizeActivationPerChannel
from enum import Enum
from itertools import count
import os
import numpy as np



BN_OUTPUT = 3


class Conv2dCust(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dCust, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, input):
        weight_q = QuantizeWeightsPerChannel(self.weight)

        weight_a = self.weight.data
        self.weight.data = weight_q
        out = super(Conv2dCust, self).forward(input)

        if out.shape[1] != 1000:
            out = QuantizeActivationPerChannel(out)
        else:
            self.weight.data = weight_a
            return out

        # Get E(x)
        global BN_OUTPUT

        if isinstance(BN_OUTPUT, list) and len(BN_OUTPUT) > 1:
            from scipy.stats import norm
            gamma = BN_OUTPUT[1].cpu().numpy()
            beta = BN_OUTPUT[0].cpu().numpy()
            cdf = norm.cdf(-beta/gamma)
            pdf = norm(0,1).pdf(-beta/gamma)
            E_x = gamma*pdf + beta*(np.ones(np.shape(gamma)[0]) - cdf)

            # Calculate E-quantization error
            bias_q = weight_q.view(weight_q.shape[0], -1).sum(1)
            bias_orig = weight_a.view(weight_a.shape[0], -1).sum(1)
            
            # Calculate epsilon
            epsilon = weight_q - weight_a
            epsilon = epsilon.view(epsilon.shape[0], epsilon.shape[1], -1).sum(2)
            if epsilon.shape[1] == 1:
                epsilon = epsilon.cpu().numpy()
                epsilon = np.repeat(epsilon, np.size(E_x), axis=1)
                epsilon = torch.tensor(epsilon)
            bias_correction = torch.matmul(epsilon.to("cuda"), torch.tensor(E_x, dtype=torch.float).to("cuda"))
            bias_correction = bias_correction/np.size(E_x)

            np_bias_c = bias_correction.cpu().numpy()
            np_bias_c = np.repeat(np_bias_c[np.newaxis, :], out.shape[0], axis=0)
            np_bias_c = np.repeat(np_bias_c[:, :, np.newaxis], out.size()[2] , axis=2)
            np_bias_c = np.repeat(np_bias_c[:, :, :, np.newaxis], out.size()[3] , axis=3)
            bias_correction = torch.from_numpy(np_bias_c).float().cuda()
            out = out - bias_correction

        #weight_a = self.weight.data
        self.weight.data = weight_a

        return out


class BatchNorm2dCust(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2dCust, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        if hasattr(self, 'absorbed'):
            global BN_OUTPUT

            if hasattr(self, 'pre_relu') and self.running_mean.shape[0] != 1280:
                BN_OUTPUT = [self.running_mean, self.running_var]
            else:
                BN_OUTPUT = []
                #print("Folded!!!")
            return input

        return super(BatchNorm2dCust, self).forward(input)

class QuantizationManagerInference():
    def __init__(self):
        self.enabled = False
        self.origin_conv2d = nn.Conv2d
        self.origin_batch_norm = nn.BatchNorm2d

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()

    def enable(self):
        nn.Conv2d = Conv2dCust
        nn.BatchNorm2d = BatchNorm2dCust
        print("Enable hit!!")

    def disable(self):
        nn.Conv2d = self.origin_conv2d
        nn.BatchNorm2d = self.origin_batch_norm
        print("Disable hit!!")
        
# Alias
QMI = QuantizationManagerInference
