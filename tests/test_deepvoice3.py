# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import sys
from os.path import dirname, join, exists

from deepvoice3_pytorch.frontend.en import text_to_sequence, n_vocab

import torch
from torch import nn
import numpy as np

from nose.plugins.attrib import attr

from deepvoice3_pytorch.builder import deepvoice3
from deepvoice3_pytorch import MultiSpeakerTTSModel, AttentionSeq2Seq


use_cuda = torch.cuda.is_available() and False
torch.backends.cudnn.deterministic = True
num_mels = 80
num_freq = 513
outputs_per_step = 4
padding_idx = 0


def _get_model(n_speakers=1, speaker_embed_dim=None,
               force_monotonic_attention=False,
               use_decoder_state_for_postnet_input=False, use_memory_mask=False):
    model = deepvoice3(n_vocab=n_vocab,
                       embed_dim=32,
                       mel_dim=num_mels,
                       linear_dim=num_freq,
                       r=outputs_per_step,
                       padding_idx=padding_idx,
                       n_speakers=n_speakers,
                       speaker_embed_dim=speaker_embed_dim,
                       dropout=1 - 0.95,
                       kernel_size=5,
                       encoder_channels=16,
                       decoder_channels=32,
                       converter_channels=32,
                       force_monotonic_attention=force_monotonic_attention,
                       use_decoder_state_for_postnet_input=use_decoder_state_for_postnet_input,
                       use_memory_mask=use_memory_mask,
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
    x = torch.LongTensor(seqs)
    y = torch.rand(x.size(0), 12, 80)

    return x, y, input_lengths


def _deepvoice3(n_vocab, embed_dim=256, mel_dim=80,
                linear_dim=4096, r=5,
                n_speakers=1, speaker_embed_dim=16,
                padding_idx=None,
                dropout=(1 - 0.95), dilation=1):

    from deepvoice3_pytorch.deepvoice3 import Encoder, Decoder, Converter
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
        preattention=[(h, 3, 1)],
        convolutions=[(h, 3, dilation), (h, 3, dilation), (h, 3, dilation),
                      (h, 3, dilation), (h, 3, dilation)],
        attention=[True, False, False, False, True],
        force_monotonic_attention=False)

    seq2seq = AttentionSeq2Seq(encoder, decoder)

    in_dim = mel_dim
    h = 256
    converter = Converter(n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim,
                          in_dim=in_dim, out_dim=linear_dim, dropout=dropout,
                          convolutions=[(h, 3, dilation), (h, 3, dilation), (h, 3, dilation),
                                        (h, 3, dilation), (h, 3, dilation)])

    model = MultiSpeakerTTSModel(
        seq2seq, converter, padding_idx=padding_idx,
        mel_dim=mel_dim, linear_dim=linear_dim,
        n_speakers=n_speakers, speaker_embed_dim=speaker_embed_dim)

    return model


def test_single_speaker_deepvoice3():
    x, y, lengths = _test_data()

    for v in [False, True]:
        model = _get_model(use_decoder_state_for_postnet_input=v)
        mel_outputs, linear_outputs, alignments, done = model(x, y, input_lengths=lengths)

    model = _get_model(use_memory_mask=True)
    mel_outputs, linear_outputs, alignments, done = model(x, y, input_lengths=lengths)


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
    x = torch.LongTensor(seqs)
    y = torch.rand(x.size(0), 4 * 33, 80)
    model = _get_model(n_speakers=32, speaker_embed_dim=16)
    speaker_ids = torch.LongTensor([1, 2, 3])

    mel_outputs, linear_outputs, alignments, done = model(x, y, speaker_ids=speaker_ids)
    print("Input text:", x.size())
    print("Input mel:", y.size())
    print("Mel:", mel_outputs.size())
    print("Linear:", linear_outputs.size())
    print("Alignments:", alignments.size())
    print("Done:", done.size())


@attr("issue38")
def test_incremental_path_multiple_times():
    texts = ["they discarded this for a more completely Roman and far less beautiful letter."]
    seqs = np.array([text_to_sequence(t) for t in texts])
    text_positions = np.arange(1, len(seqs[0]) + 1).reshape(1, len(seqs[0]))

    r = 4
    mel_dim = 80
    sequence = torch.LongTensor(seqs)
    text_positions = torch.LongTensor(text_positions)

    for model, speaker_ids in [
            (_get_model(force_monotonic_attention=False), None),
            (_get_model(force_monotonic_attention=False, n_speakers=32, speaker_embed_dim=16), torch.LongTensor([1]))]:
        model.eval()

        # first call
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

        # second call
        mel_outputs2, linear_outputs2, alignments2, done2 = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

        # Should get same result
        c = (mel_outputs - mel_outputs2).abs()
        print(c.mean(), c.max())

        assert np.allclose(mel_outputs.cpu().data.numpy(),
                           mel_outputs2.cpu().data.numpy(), atol=1e-5)


def test_incremental_correctness():
    texts = ["they discarded this for a more completely Roman and far less beautiful letter."]
    seqs = np.array([text_to_sequence(t) for t in texts])
    text_positions = np.arange(1, len(seqs[0]) + 1).reshape(1, len(seqs[0]))

    mel_path = join(dirname(__file__), "data", "ljspeech-mel-00001.npy")
    mel = np.load(mel_path)
    max_target_len = mel.shape[0]
    r = 4
    mel_dim = 80
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    mel = _pad_2d(mel, max_target_len)
    mel = torch.from_numpy(mel)
    mel_reshaped = mel.contiguous().view(1, -1, mel_dim * r)
    frame_positions = np.arange(1, mel_reshaped.size(1) + 1).reshape(1, mel_reshaped.size(1))

    x = torch.LongTensor(seqs)
    text_positions = torch.LongTensor(text_positions)
    frame_positions = torch.LongTensor(frame_positions)

    for model, speaker_ids in [
            (_get_model(force_monotonic_attention=False), None),
            (_get_model(force_monotonic_attention=False, n_speakers=32, speaker_embed_dim=16), torch.LongTensor([1]))]:
        model.eval()

        if speaker_ids is not None:
            speaker_embed = model.embed_speakers(speaker_ids)
        else:
            speaker_embed = None

        # Encoder
        encoder_outs = model.seq2seq.encoder(x, speaker_embed=speaker_embed)

        # Off line decoding
        mel_outputs_offline, alignments_offline, done, _ = model.seq2seq.decoder(
            encoder_outs, mel_reshaped, speaker_embed=speaker_embed,
            text_positions=text_positions, frame_positions=frame_positions)

        # Online decoding with test inputs
        model.seq2seq.decoder.start_fresh_sequence()
        mel_outputs_online, alignments, dones_online, _ = model.seq2seq.decoder.incremental_forward(
            encoder_outs, text_positions, speaker_embed=speaker_embed,
            test_inputs=mel_reshaped)

        # Should get same result
        c = (mel_outputs_offline - mel_outputs_online).abs()
        print(c.mean(), c.max())

        assert np.allclose(mel_outputs_offline.cpu().data.numpy(),
                           mel_outputs_online.cpu().data.numpy(), atol=1e-5)


@attr("local_only")
def test_incremental_forward():
    checkpoint_path = join(dirname(__file__), "../test_whole/checkpoint_step000265000.pth")
    if not exists(checkpoint_path):
        return
    model = _get_model()

    use_cuda = False

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.make_generation_fast_()
    model = model.cuda() if use_cuda else model

    texts = ["they discarded this for a more completely Roman and far less beautiful letter."]
    seqs = np.array([text_to_sequence(t) for t in texts])
    input_lengths = [len(s) for s in seqs]

    use_manual_padding = False
    if use_manual_padding:
        max_input_len = np.max(input_lengths) + 10  # manuall padding
        seqs = np.array([_pad(x, max_input_len) for x in seqs], dtype=np.int)
        input_lengths = torch.LongTensor(input_lengths)
        input_lengths = input_lengths.cuda() if use_cuda else input_lengths
    else:
        input_lengths = None

    text_positions = np.arange(1, len(seqs[0]) + 1).reshape(1, len(seqs[0]))

    mel = np.load("/home/ryuichi/Dropbox/sp/deepvoice3_pytorch/data/ljspeech/ljspeech-mel-00035.npy")
    max_target_len = mel.shape[0]
    r = 4
    mel_dim = 80
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    mel = _pad_2d(mel, max_target_len)
    mel = torch.from_numpy(mel)
    mel_reshaped = mel.contiguous().view(1, -1, mel_dim * r)

    frame_positions = np.arange(1, mel_reshaped.size(1) + 1).reshape(1, mel_reshaped.size(1))

    x = torch.LongTensor(seqs)
    text_positions = torch.LongTensor(text_positions)
    frame_positions = torch.LongTensor(frame_positions)

    if use_cuda:
        x = x.cuda()
        text_positions = text_positions.cuda()
        frame_positions = frame_positions.cuda()
        mel_reshaped = mel_reshaped.cuda()

    model.eval()

    def _plot(mel, mel_predicted, alignments):
        from matplotlib import pylab as plt
        plt.figure(figsize=(16, 10))
        plt.subplot(3, 1, 1)
        plt.imshow(mel.data.cpu().numpy().T, origin="lower bottom",
                   aspect="auto", cmap="magma")
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

    # Encoder
    encoder_outs = model.seq2seq.encoder(x, lengths=input_lengths)

    # Off line decoding
    mel_output_offline, alignments_offline, done = model.seq2seq.decoder(
        encoder_outs, mel_reshaped,
        text_positions=text_positions, frame_positions=frame_positions,
        lengths=input_lengths)

    _plot(mel, mel_output_offline, alignments_offline)

    # Online decoding
    test_inputs = None
    # test_inputs = mel_reshaped
    model.seq2seq.decoder.start_fresh_sequence()
    mel_outputs, alignments, dones_online = model.seq2seq.decoder.incremental_forward(
        encoder_outs, text_positions,
        # initial_input=mel_reshaped[:, :1, :],
        test_inputs=test_inputs)

    if test_inputs is not None:
        c = (mel_output_offline - mel_outputs).abs()
        print(c.mean(), c.max())
        _plot(mel, c, alignments)

    _plot(mel, mel_outputs, alignments)
