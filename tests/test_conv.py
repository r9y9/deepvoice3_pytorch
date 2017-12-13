# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from deepvoice3_pytorch.conv import Conv1d


def test_conv1d_incremental():
    def __test(kernel_size, dilation, T, B, C, causual=True):
        kernel_size = 3
        dilation = (dilation,)

        # dilation = (4,)
        # causual
        assert causual
        if causual:
            padding = (kernel_size - 1) * dilation[0]
        else:
            padding = (kernel_size - 1) // 2 * dilation[0]

        # weight: (Cout, Cin, K)
        conv = nn.Conv1d(
            C, C * 2, kernel_size=kernel_size, padding=padding,
            dilation=dilation).eval()
        conv.weight.data.fill_(1.0)
        conv.bias.data.zero_()

        # weight: (K, Cin, Cout)
        # weight (linearized): (Cout*K, Cin)
        conv_online = Conv1d(
            C, C * 2, kernel_size=kernel_size, padding=padding,
            dilation=dilation).eval()
        conv_online.weight.data.fill_(1.0)
        conv_online.bias.data.zero_()

        # (B, C, T)
        bct = Variable(torch.zeros(B, C, T) + torch.arange(0, T))
        output_conv = conv(bct)

        # Remove future time stamps
        output_conv = output_conv[:, :, :T]

        output_conv_online = []

        # B, T, C
        btc = bct.transpose(1, 2).contiguous()
        for t in range(btc.size(1)):
            input = btc[:, t, :].contiguous().view(B, -1, C)
            output = conv_online.incremental_forward(input)
            output_conv_online += [output]

        output_conv_online = torch.stack(output_conv_online).squeeze(2)
        output_conv_online = output_conv_online.transpose(0, 1).transpose(1, 2)

        assert (output_conv == output_conv_online).all()

    for B in [1, 16]:
        for T in [10, 20, 30]:
            for C in [1, 2, 4]:
                for kernel_size in [3, 5, 9]:
                    for dilation in [1, 2, 3, 4, 5, 6, 7, 8, 9, 27]:
                        __test, kernel_size, dilation, T, B, C
