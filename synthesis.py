# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>        Hyper parameters [default: ].
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join

import audio
from train import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend, build_deepvoice3
from hparams import hparams

from tqdm import tqdm

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later


def tts(model, text, p=0):
    """Convert text to speech waveform given a deepvoice3 model.
    """
    if use_cuda:
        model = model.cuda()
    model.eval()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    if use_cuda:
        sequence = sequence.cuda()
        text_positions = text_positions.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, done = model(
        sequence, text_positions=text_positions)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)

    # Model
    model = build_deepvoice3(n_vocab=_frontend.n_vocab,
                             embed_dim=256,
                             mel_dim=hparams.num_mels,
                             linear_dim=hparams.num_freq,
                             r=hparams.outputs_per_step,
                             padding_idx=hparams.padding_idx,
                             dropout=hparams.dropout,
                             kernel_size=hparams.kernel_size,
                             encoder_channels=hparams.encoder_channels,
                             decoder_channels=hparams.decoder_channels,
                             converter_channels=hparams.converter_channels,
                             )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps
    model.make_generation_fast_()

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            # print("{}: {} ({} chars, {} words)".format(idx, text, len(text), len(words)))
            waveform, alignment, _, _ = tts(model, text)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(idx, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}{}_alignment.png".format(idx, file_name_suffix))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="deepvoice3, {}".format(checkpoint_path))
            audio.save_wav(waveform, dst_wav_path)
            from os.path import basename, splitext
            name = splitext(basename(text_list_file_path))[0]
            print("""
{}

({} chars, {} words)

<audio controls="controls" >
<source src="/audio/deepvoice3/{}/{}{}.wav" autoplay/>
Your browser does not support the audio element.
</audio>

<div align="center"><img src="/audio/deepvoice3/{}/{}{}_alignment.png" /></div>
                  """.format(text, len(text), len(words),
                             name, idx, file_name_suffix,
                             name, idx, file_name_suffix))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
