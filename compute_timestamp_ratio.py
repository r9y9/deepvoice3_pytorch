"""Compute output/input timestamp ratio.

usage: compute_timestamp_ratio.py [options] <data_root>

options:
    --hparams=<parmas>        Hyper parameters [default: ].
    --preset=<json>           Path of preset parameters (json).
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
    data_root = args["<data_root>"]
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    train._frontend = getattr(frontend, hparams.frontend)

    # Code below
    X = FileSourceDataset(TextDataSource(data_root))
    Mel = FileSourceDataset(MelSpecDataSource(data_root))

    in_sizes = []
    out_sizes = []
    for i in trange(len(X)):
        x, m = X[i], Mel[i]
        if X.file_data_source.multi_speaker:
            x = x[0]
        in_sizes.append(x.shape[0])
        out_sizes.append(m.shape[0])

    in_sizes = np.array(in_sizes)
    out_sizes = np.array(out_sizes)

    input_timestamps = np.sum(in_sizes)
    output_timestamps = np.sum(out_sizes) / hparams.outputs_per_step / hparams.downsample_step

    print(input_timestamps, output_timestamps, output_timestamps / input_timestamps)
    sys.exit(0)
