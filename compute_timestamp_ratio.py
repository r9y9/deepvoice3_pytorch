"""Compute output/input timestamp ratio.

usage: compute_timestamp_ratio.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features.
    --hparams=<parmas>        Hyper parameters [default: ].
    -h, --help                Show this help message and exit
"""
from docopt import docopt
import sys
import numpy as np
from hparams import hparams, hparams_debug_string
import train
from train import TextDataSource, MelSpecDataSource
from nnmnkwii.datasets import FileSourceDataset
from tqdm import trange
from deepvoice3_pytorch import frontend

if __name__ == "__main__":
    args = docopt(__doc__)
    data_root = args["--data-root"]
    if data_root:
        train.DATA_ROOT = data_root
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    train._frontend = getattr(frontend, hparams.frontend)

    # Code below
    X = FileSourceDataset(TextDataSource())
    Mel = FileSourceDataset(MelSpecDataSource())

    in_sizes = []
    out_sizes = []
    for i in trange(len(X)):
        x, m = X[i], Mel[i]
        in_sizes.append(x.shape[0])
        out_sizes.append(m.shape[0])

    in_sizes = np.array(in_sizes)
    out_sizes = np.array(out_sizes)

    input_timestamps = np.sum(in_sizes)
    output_timestamps = np.sum(out_sizes) / hparams.outputs_per_step

    print(input_timestamps, output_timestamps, output_timestamps / input_timestamps)

    sys.exit(0)
