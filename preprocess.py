# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <name> <in_dir> <out_dir>

options:
    --num_workers=<n>        Num workers.
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from multiprocessing import cpu_count
from tqdm import tqdm
import importlib
from hparams import hparams, hparams_debug_string


def preprocess(mod, in_dir, out_root, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = mod.build_from_path(in_dir, out_dir, num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    frame_shift_ms = hparams.hop_size / hparams.sample_rate * 1000
    hours = frames * frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


if __name__ == "__main__":
    args = docopt(__doc__)
    name = args["<name>"]
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    num_workers = args["--num_workers"]
    num_workers = cpu_count() if num_workers is None else int(num_workers)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"
    print(hparams_debug_string())

    assert name in ["jsut", "ljspeech", "vctk", "nikl_m", "nikl_s"]
    mod = importlib.import_module(name)
    preprocess(mod, in_dir, out_dir, num_workers)
