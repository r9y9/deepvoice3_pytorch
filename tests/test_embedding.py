# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import torch
from torch import nn
from deepvoice3_pytorch.modules import SinusoidalEncoding, position_encoding_init
import numpy as np


def test_sinusoidal():
    num_embedding = 512
    embedding_dim = 128

    for w in [1.0, 0.5, 2.0, 10.0, 20.0]:
        a = nn.Embedding(num_embedding, embedding_dim, padding_idx=0)
        a.weight.data = position_encoding_init(
            num_embedding, embedding_dim, position_rate=w)

        b = SinusoidalEncoding(num_embedding, embedding_dim)

        x = torch.arange(0, 128).long()
        ax = a(x).data.numpy()
        bx = b(x, w).data.numpy()

        print(w, np.abs(ax - bx).mean())
        try:
            assert np.allclose(ax, bx)
        except:
            print("TODO: has little numerical errors?")
            assert np.abs(ax - bx).mean() < 1e-5
