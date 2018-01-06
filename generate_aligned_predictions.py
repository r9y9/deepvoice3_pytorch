# coding: utf-8
"""
Generate ground trouth-aligned predictions

usage: generate_aligned_predictions.py [options] <checkpoint> <in_dir> <out_dir>

options:
    --hparams=<parmas>       Hyper parameters [default: ].
    --overwrite              Overwrite audio and mel outputs.
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from tqdm import tqdm
import importlib
from os.path import join
from warnings import warn
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later


def preprocess(model, in_dir, out_dir, text, audio_filename, mel_filename,
               p=0, speaker_id=None,
               fast=False):
    """Generate ground truth-aligned prediction

    The output of the network and corresponding audio are saved after time
    resolution adjustment.
    """
    r = hparams.outputs_per_step
    downsample_step = hparams.downsample_step

    if use_cuda:
        model = model.cuda()
    model.eval()
    if fast:
        model.make_generation_fast_()

    mel_org = np.load(join(in_dir, mel_filename))
    mel = Variable(torch.from_numpy(mel_org)).unsqueeze(0).contiguous()

    # Downsample mel spectrogram
    if downsample_step > 1:
        mel = mel[:, 0::downsample_step, :].contiguous()

    decoder_target_len = mel.shape[1] // r
    s, e = 1, decoder_target_len + 1
    frame_positions = torch.arange(s, e).long().unsqueeze(0)
    frame_positions = Variable(frame_positions)

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))
    if use_cuda:
        sequence = sequence.cuda()
        text_positions = text_positions.cuda()
        speaker_ids = None if speaker_ids is None else speaker_ids.cuda()
        mel = mel.cuda()
        frame_positions = frame_positions.cuda()

    # **Teacher forcing** decoding
    mel_outputs, _, _, _ = model(
        sequence, mel, text_positions=text_positions,
        frame_positions=frame_positions, speaker_ids=speaker_ids)

    mel_output = mel_outputs[0].data.cpu().numpy()

    # **Time resolution adjustment**
    # remove begenning audio used for first mel prediction
    wav = np.load(join(in_dir, audio_filename))[hparams.hop_size * downsample_step:]
    assert len(wav) % hparams.hop_size == 0

    # Coarse upsample just for convenience
    # so that we can upsample conditional features by hop_size in wavenet
    if downsample_step > 0:
        mel_output = np.repeat(mel_output, downsample_step, axis=0)
    # downsampling -> upsampling, then we should have length equal to or larger than
    # the original mel length
    assert mel_output.shape[0] >= mel_org.shape[0]

    # Trim mel output
    expected_frames = len(wav) // hparams.hop_size
    mel_output = mel_output[:expected_frames]

    # Make sure we have correct lengths
    assert mel_output.shape[0] * hparams.hop_size == len(wav)

    timesteps = len(wav)

    # save
    np.save(join(out_dir, audio_filename), wav.astype(np.int16),
            allow_pickle=False)
    np.save(join(out_dir, mel_filename), mel_output.astype(np.float32),
            allow_pickle=False)

    if speaker_id is None:
        return (audio_filename, mel_filename, timesteps, text)
    else:
        return (audio_filename, mel_filename, timesteps, text, speaker_id)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / sr / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    args = docopt(__doc__)
    checkpoint_path = args["<checkpoint>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import build_model

    model = build_model()

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    os.makedirs(out_dir, exist_ok=True)
    results = []
    with open(os.path.join(in_dir, "train.txt")) as f:
        lines = f.readlines()

    for idx in tqdm(range(len(lines))):
        l = lines[idx]
        l = l[:-1].split("|")
        audio_filename, mel_filename, _, text = l[:4]
        speaker_id = int(l[4]) if len(l) > 4 else None
        if text == "N/A":
            raise RuntimeError("No transcription available")

        result = preprocess(model, in_dir, out_dir, text, audio_filename,
                            mel_filename, p=0,
                            speaker_id=speaker_id, fast=True)
        results.append(result)

    write_metadata(results, out_dir)

    sys.exit(0)
