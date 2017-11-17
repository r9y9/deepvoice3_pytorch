# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import sys
from os.path import dirname, join, exists

from deepvoice3_pytorch.frontend.en import text_to_sequence, n_vocab

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

from nose.plugins.attrib import attr

from deepvoice3_pytorch.builder import build_nyanko
from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq

from fairseq.modules.conv_tbc import ConvTBC

use_cuda = torch.cuda.is_available()
num_mels = 80
num_freq = 513
outputs_per_step = 4
padding_idx = 0


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _test_data():
    texts = ["Thank you very much.", "Hello.", "Deep voice 3."]
    seqs = [np.array(text_to_sequence(t), dtype=np.int) for t in texts]
    input_lengths = np.array([len(s) for s in seqs])
    max_len = np.max(input_lengths)
    seqs = np.array([_pad(s, max_len) for s in seqs])

    # Test encoder
    x = Variable(torch.LongTensor(seqs))
    y = Variable(torch.rand(x.size(0), 12, 80))

    return x, y


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


@attr("local_only")
def test_nyanko():
    texts = ["they discarded this for a more completely Roman and far less beautiful letter."]
    seqs = np.array([text_to_sequence(t) for t in texts])
    text_positions = np.arange(1, len(seqs[0]) + 1).reshape(1, len(seqs[0]))

    mel = np.load("/home/ryuichi/Dropbox/sp/deepvoice3_pytorch/data/ljspeech/ljspeech-mel-00035.npy")
    max_target_len = mel.shape[0]
    r = 1
    mel_dim = 80
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    mel = _pad_2d(mel, max_target_len)
    mel = Variable(torch.from_numpy(mel))
    mel_reshaped = mel.view(1, -1, mel_dim * r)
    frame_positions = np.arange(1, mel_reshaped.size(1) + 1).reshape(1, mel_reshaped.size(1))

    x = Variable(torch.LongTensor(seqs))
    text_positions = Variable(torch.LongTensor(text_positions))
    frame_positions = Variable(torch.LongTensor(frame_positions))

    model = build_nyanko(n_vocab, mel_dim=mel_dim, linear_dim=513,
                         r=r, force_monotonic_attention=False)
    model.eval()

    def _plot(mel, mel_predicted, alignments):
        from matplotlib import pylab as plt
        plt.figure(figsize=(16, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(mel.data.cpu().numpy().T, origin="lower bottom", aspect="auto", cmap="magma")
        plt.colorbar()

        plt.subplot(3, 1, 2)
        plt.imshow(mel_predicted.view(-1, mel_dim).data.cpu().numpy().T,
                   origin="lower bottom", aspect="auto", cmap="magma")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        if alignments.dim() == 4:
            alignments = alignments.mean(0)
        plt.imshow(alignments[0].data.cpu(
        ).numpy().T, origin="lower bottom", aspect="auto")
        plt.colorbar()
        plt.show()

    seq2seq = model.seq2seq

    # Encoder
    encoder_outs = seq2seq.encoder(x)

    # Off line decoding
    print("Offline decoding")
    mel_outputs_offline, alignments_offline, done = seq2seq.decoder(
        encoder_outs, mel_reshaped,
        text_positions=text_positions, frame_positions=frame_positions)

    # Online decoding with test inputs
    print("Online decoding")
    seq2seq.decoder._start_incremental_inference()
    mel_outputs_online, alignments, dones_online = seq2seq.decoder._incremental_forward(
        encoder_outs, text_positions,
        test_inputs=mel_reshaped)
    seq2seq.decoder._stop_incremental_inference()

    a = mel_outputs_offline.cpu().data.numpy()
    b = mel_outputs_online.cpu().data.numpy()
    c = (mel_outputs_offline - mel_outputs_online).abs()
    print(c.mean(), c.max())

    _plot(mel, mel_outputs_offline, alignments_offline)
    _plot(mel, mel_outputs_online, alignments)
    _plot(mel, c, alignments)

    import ipdb
    ipdb.set_trace()
    # Should get same result
    # TODO
    if False:
        assert np.allclose(a, b)

    postnet = model.postnet

    linear_outputs = postnet(mel_outputs_offline)
    print(linear_outputs.size())
    import ipdb
    ipdb.set_trace()
