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

from deepvoice3_pytorch import Encoder, Decoder, Converter, DeepVoice3
from deepvoice3_pytorch import build_deepvoice3

from fairseq.modules.conv_tbc import ConvTBC

use_cuda = torch.cuda.is_available()
num_mels = 80
num_freq = 1025
outputs_per_step = 4
padding_idx = 0


def _get_model(n_speakers=1, speaker_embed_dim=None):
    model = build_deepvoice3(n_vocab=n_vocab,
                             embed_dim=256,
                             mel_dim=num_mels,
                             linear_dim=num_freq,
                             r=outputs_per_step,
                             padding_idx=padding_idx,
                             n_speakers=n_speakers,
                             speaker_embed_dim=speaker_embed_dim,
                             )
    return model


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


def _build_deepvoice3(n_vocab, embed_dim=256, mel_dim=80,
                      linear_dim=4096, r=5,
                      n_speakers=1, speaker_embed_dim=16,
                      padding_idx=None,
                      dropout=(1 - 0.95), dilation=1):
    h = 128
    encoder = Encoder(
        n_vocab, embed_dim, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout,
        convolutions=[(h, 3, dilation), (h, 3, dilation), (h, 3, dilation),
                      (h, 3, dilation), (h, 3, dilation)],
    )

    h = 256
    decoder = Decoder(
        embed_dim, in_dim=mel_dim, r=r, padding_idx=padding_idx,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
        dropout=dropout,
        convolutions=[(h, 3, dilation), (h, 3, dilation), (h, 3, dilation),
                      (h, 3, dilation), (h, 3, dilation)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=[True, False, False, False, False])

    in_dim = h // r
    h = 256
    converter = Converter(
        in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
        convolutions=[(h, 3, dilation), (h, 3, dilation), (h, 3, dilation),
                      (h, 3, dilation), (h, 3, dilation)])

    model = DeepVoice3(
        encoder, decoder, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim)

    return model


def test_single_speaker_deepvoice3():
    x, y = _test_data()

    model = _get_model()
    mel_outputs, linear_outputs, alignments, done = model(x, y)


def test_dilated_convolution_support():
    x, y = _test_data()

    for dilation in [1, 2]:
        model = _build_deepvoice3(n_vocab=n_vocab,
                                  embed_dim=256,
                                  mel_dim=num_mels,
                                  linear_dim=num_freq,
                                  r=outputs_per_step,
                                  padding_idx=padding_idx,
                                  n_speakers=1,
                                  speaker_embed_dim=16,
                                  dilation=dilation,
                                  )
        if dilation > 1:
            for conv in [model.encoder.convolutions[0],
                         model.decoder.convolutions[0],
                         model.converter.convolutions[0]]:
                assert isinstance(conv, nn.Conv1d)
        else:
            assert dilation == 1
            for conv in [model.encoder.convolutions[0],
                         model.decoder.convolutions[0],
                         model.converter.convolutions[0]]:
                assert isinstance(conv, ConvTBC)
        mel_outputs, linear_outputs, alignments, done = model(x, y)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


def test_multi_speaker_deepvoice3():
    texts = ["Thank you very much.", "Hello.", "Deep voice 3."]
    seqs = [np.array(text_to_sequence(t), dtype=np.int) for t in texts]
    input_lengths = np.array([len(s) for s in seqs])
    max_len = np.max(input_lengths)
    seqs = np.array([_pad(s, max_len) for s in seqs])

    # Test encoder
    x = Variable(torch.LongTensor(seqs))
    y = Variable(torch.rand(x.size(0), 4 * 33, 80))
    model = _get_model(n_speakers=32, speaker_embed_dim=16)
    speaker_ids = Variable(torch.LongTensor([1, 2, 3]))

    mel_outputs, linear_outputs, alignments, done = model(x, y, speaker_ids=speaker_ids)
    print("Input text:", x.size())
    print("Input mel:", y.size())
    print("Mel:", mel_outputs.size())
    print("Linear:", linear_outputs.size())
    print("Alignments:", alignments.size())
    print("Done:", done.size())


@attr("local_only")
def test_incremental_forward():
    checkpoint_path = join(dirname(__file__), "../checkpoints/checkpoint_step000140000.pth")
    if not exists(checkpoint_path):
        return
    model = _get_model()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda() if use_cuda else model

    texts = ["they discarded this for a more completely Roman and far less beautiful letter."]
    seqs = np.array([text_to_sequence(t) for t in texts])
    input_lengths = [len(s) for s in seqs]

    use_manual_padding = False
    if use_manual_padding:
        max_input_len = np.max(input_lengths) + 10  # manuall padding
        seqs = np.array([_pad(x, max_input_len) for x in seqs], dtype=np.int)
        input_lengths = torch.LongTensor(input_lengths)
        input_lengths = input_lengths.cuda() if use_cuda else input_lenghts
    else:
        input_lengths = None

    text_positions = np.arange(1, len(seqs[0]) + 1).reshape(1, len(seqs[0]))

    mel = np.load("/home/ryuichi/tacotron/training/ljspeech-mel-00035.npy")
    max_target_len = mel.shape[0]
    r = 4
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

    if use_cuda:
        x = x.cuda()
        text_positions = text_positions.cuda()
        frame_positions = frame_positions.cuda()
        mel_reshaped = mel_reshaped.cuda()

    # model.make_generation_fast_()
    model.eval()

    encoder_outs = model.encoder(x, lengths=input_lengths)

    # Off line decoding
    mel_output_offline, alignments_offline, done, decoder_states = model.decoder(
        encoder_outs, mel_reshaped,
        text_positions=text_positions, frame_positions=frame_positions,
        lengths=input_lengths)

    from matplotlib import pylab as plt

    def _plot(mel, mel_predicted, alignments):
        plt.figure(figsize=(16, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(mel.data.cpu().numpy().T, origin="lower bottom", aspect="auto")
        plt.colorbar()

        plt.subplot(3, 1, 2)
        plt.imshow(mel_predicted.view(-1, mel_dim).data.cpu().numpy().T,
                   origin="lower bottom", aspect="auto")
        plt.colorbar()

        plt.subplot(3, 1, 3)
        if alignments.dim() == 4:
            alignments = alignments.mean(0)
        plt.imshow(alignments[0].data.cpu(
        ).numpy().T, origin="lower bottom", aspect="auto")
        plt.colorbar()
        plt.show()

    _plot(mel, mel_output_offline, alignments_offline)

    # Online decoding
    model.decoder._start_incremental_inference()
    mel_outputs, alignments, dones_online, decoder_states_online = model.decoder._incremental_forward(
        encoder_outs, text_positions,
        # initial_input=mel_reshaped[:, :1, :],
        test_inputs=None)
    # test_inputs=mel_reshaped)
    model.decoder._stop_incremental_inference()

    _plot(mel, mel_outputs, alignments)
