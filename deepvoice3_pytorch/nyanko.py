# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import numpy as np

from fairseq.models.fconv import Embedding, Linear


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    from .conv import Conv1d
    m = Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class Converter(nn.Module):
    def __init__(self, in_dim, out_dim,
                 convolutions=((256, 5, 1),) * 6,
                 deconvolutions=((256, 5, 1),) * 2,  # do upsampling
                 dropout=0.1):
        super(Converter, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Non-causual convolutions
        in_channels = convolutions[0][0]
        self.fc1 = Linear(in_dim, in_channels)

        # Convlutions
        self.convolutions = nn.ModuleList()
        self.deconvolutions = nn.ModuleList()
        for idx, (out_channels, kernel_size, dilation) in enumerate(convolutions):
            if idx < len(deconvolutions):
                self.deconvolutions.append(
                    ConvTranspose1d(in_channels, out_channels, kernel_size=2,
                                    padding=0, stride=2))
            pad = (kernel_size - 1) // 2 * dilation
            dilation = (dilation,)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size,
                       padding=pad, dilation=dilation, dropout=dropout))
            in_channels = out_channels

        self.fc2 = Linear(in_channels, out_dim)

    def forward(self, x):
        # project to size of convolution
        x = self.fc1(x)

        # TBC case: B x T x C -> T x B x C
        # Generic case: B x T x C -> B x C x T
        use_convtbc = False
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        # Conv blocks
        for idx, conv in enumerate(self.convolutions):
            # Upsampling
            if idx < len(self.deconvolutions):
                x = self.deconvolutions[idx](x)
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            splitdim = -1 if use_convtbc else 1
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            x = a * F.sigmoid(b)
            x = (x + residual) * math.sqrt(0.5)

        # Back to batch first
        x = x.transpose(0, 1) if use_convtbc else x.transpose(1, 2)

        return F.sigmoid(self.fc2(x))
