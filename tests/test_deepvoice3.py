# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import sys
from os.path import dirname, join
tacotron_lib_dir = join(dirname(__file__), "..", "lib", "tacotron")
sys.path.append(tacotron_lib_dir)
from text import text_to_sequence, symbols

import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

from deepvoice3_pytorch import Encoder, DeepVoice3, build_deepvoice3


def test_model():
    assert True


def _pad(seq, max_len):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def test_single_spaker_deepvoice3():
    texts = ["Thank you very much.", "Hello.", "Deep voice 3."]
    seqs = [np.array(text_to_sequence(
        t, ["english_cleaners"]), dtype=np.int) for t in texts]
    input_lengths = np.array([len(s) for s in seqs])
    max_len = np.max(input_lengths)
    seqs = np.array([_pad(s, max_len) for s in seqs])

    # Test encoder
    x = Variable(torch.LongTensor(seqs))
    y = Variable(torch.rand(x.size(0), 10, 80))
    model = build_deepvoice3(n_vocab=len(symbols))

    mel_outputs, linear_outputs, alignments = model(x, y)


def test_multi_speaker_deepvoice3():
    texts = ["Thank you very much.", "Hello.", "Deep voice 3."]
    seqs = [np.array(text_to_sequence(
        t, ["english_cleaners"]), dtype=np.int) for t in texts]
    input_lengths = np.array([len(s) for s in seqs])
    max_len = np.max(input_lengths)
    seqs = np.array([_pad(s, max_len) for s in seqs])

    # Test encoder
    x = Variable(torch.LongTensor(seqs))
    y = Variable(torch.rand(x.size(0), 5 * 33, 80))
    model = build_deepvoice3(n_vocab=len(symbols), n_speakers=32, speaker_embed_dim=16)
    speaker_ids = Variable(torch.LongTensor([1, 2, 3]))

    mel_outputs, linear_outputs, alignments = model(x, y, speaker_ids=speaker_ids)
    print("Input text:", x.size())
    print("Input mel:", y.size())
    print("Mel:", mel_outputs.size())
    print("Linear:", linear_outputs.size())
    print("Alignments:", alignments.size())
