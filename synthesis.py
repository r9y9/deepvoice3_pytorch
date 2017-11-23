# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext

import audio

import torch
from torch.autograd import Variable
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams

from tqdm import tqdm

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later


def tts(model, text, p=0):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    if use_cuda:
        model = model.cuda()
    model.eval()
    model.make_generation_fast_()

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
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    # Presets
    if hparams.use_preset:
        preset = hparams.presets[hparams.builder]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.builder, json.dumps(preset, indent=4)))

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import plot_alignment, build_model

    # Model
    model = build_model()

    # Load checkpoints separately
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = torch.load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = torch.load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            waveform, alignment, _, _ = tts(model, text, p=replace_pronunciation_prob)
            dst_wav_path = join(dst_dir, "{}_{}{}.wav".format(
                idx, checkpoint_name, file_name_suffix))
            dst_alignment_path = join(
                dst_dir, "{}_{}{}_alignment.png".format(idx, checkpoint_name,
                                                        file_name_suffix))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="{}, {}".format(hparams.builder, basename(checkpoint_path)))
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
