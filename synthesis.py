# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --checkpoint-wavenet=<path>       Load WaveNet vocoder.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
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
from hparams import hparams, hparams_debug_string

from tqdm import tqdm

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later


def tts(model, text, p=0, speaker_id=None, fast=False, wavenet=None):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    if use_cuda:
        model = model.cuda()
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0).long()
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))
    if use_cuda:
        sequence = sequence.cuda()
        text_positions = text_positions.cuda()
        speaker_ids = None if speaker_ids is None else speaker_ids.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, done = model(
        sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    if wavenet is not None:
        if use_cuda:
            wavenet = wavenet.cuda()
        wavenet.eval()
        if fast:
            wavenet.make_generation_fast_()

        # TODO: assuming scalar input
        initial_value = 0.0
        initial_input = Variable(torch.zeros(1, 1, 1)).fill_(initial_value)
        # (B, T, C) -> (B, C, T)
        c = mel_outputs.transpose(1, 2).contiguous()
        g = None
        Tc = c.size(-1)
        length = Tc * 256
        if use_cuda:
            initial_input = initial_input.cuda()
            c = c.cuda()
        waveform = wavenet.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=float(np.log(1e-14)))
        waveform = waveform.view(-1).cpu().data.numpy()
    else:
        waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    checkpoint_wavenet_path = args["--checkpoint-wavenet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    if speaker_id is not None:
        speaker_id = int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import plot_alignment, build_model

    # Model
    model = build_model()

    # Load checkpoints separately
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = _load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = _load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = _load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    # Load WaveNet vocoder
    if checkpoint_wavenet_path is not None:
        from wavenet_vocoder import builder
        wavenet = builder.wavenet(out_channels=3 * 10, layers=24, stacks=4, residual_channels=512,
                                  gate_channels=512, skip_out_channels=256, dropout=1 - 0.95,
                                  kernel_size=3, weight_normalization=True, cin_channels=80,
                                  upsample_conditional_features=True, upsample_scales=[4, 4, 4, 4],
                                  freq_axis_kernel_size=3, gin_channels=-1, scalar_input=True)
        checkpoint = torch.load(checkpoint_wavenet_path)
        wavenet.load_state_dict(checkpoint["state_dict"])
    else:
        wavenet = None

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)
    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            waveform, alignment, _, _ = tts(
                model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True,
                wavenet=wavenet)
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
            if output_html:
                print("""
{}

({} chars, {} words)

<audio controls="controls" >
<source src="/audio/{}/{}/{}" autoplay/>
Your browser does not support the audio element.
</audio>

<div align="center"><img src="/audio/{}/{}/{}" /></div>
                  """.format(text, len(text), len(words),
                             hparams.builder, name, basename(dst_wav_path),
                             hparams.builder, name, basename(dst_alignment_path)))
            else:
                print(idx, ": {}\n ({} chars, {} words)".format(text, len(text), len(words)))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
