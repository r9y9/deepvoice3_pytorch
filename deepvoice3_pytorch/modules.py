# coding: utf-8

from torch import nn
import math

from torch.nn import functional as F
from fairseq.models.fconv import Linear, LinearizedConvolution


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.01)
    return m


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=4, **kwargs):
    from .conv import Conv1d
    m = Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTranspose1d(in_channels, out_channels, kernel_size, dropout=0, std_mul=1, **kwargs):
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def LinearizedConv1d(in_channels, out_channels, kernel_size, dilation=(1,),
                     dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    assert dilation[0] == 1
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def ConvTBC(in_channels, out_channels, kernel_size, dilation=(1,),
            dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    assert dilation[0] == 1
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


class HighwayConv1d(nn.Module):
    """Conv1d + Highway network (support incremental forward)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=None,
                 dilation=1, causal=False, dropout=0, std_mul=4.0, glu=False):
        super(HighwayConv1d, self).__init__()
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal
        self.dropout = dropout
        self.glu = glu

        self.conv = Conv1d(in_channels, 2 * out_channels,
                           kernel_size=kernel_size, padding=padding,
                           dilation=dilation, dropout=dropout,
                           std_mul=std_mul)

    def forward(self, x):
        return self._forward(x, False)

    def incremental_forward(self, x):
        return self._forward(x, True)

    def _forward(self, x, is_incremental):
        """Forward

        Args:
            x: (B, in_channels, T)
        returns:
            (B, out_channels, T)
        """

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        if self.glu:
            x = F.glu(x, dim=splitdim)
            return (x + residual) * math.sqrt(0.5)
        else:
            a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
            T = F.sigmoid(b)
            # TODO: does math.sqrt(0.5) really make sense?
            # This helps training stability
            return (T * a + (1 - T) * residual) * math.sqrt(0.5)

    def clear_buffer(self):
        self.conv.clear_buffer()


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask
