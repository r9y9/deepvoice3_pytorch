# coding: utf-8
"""
Script for do force alignment by gentle for VCTK. This script takes approx
~40 hours to finish. It processes all utterances in VCTK.

NOTE: Must be run with Python2, since gentle doesn't work with Python3.

Usage:
    1. Install https://github.com/lowerquality/gentle
    2. Download VCTK http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html

and then run the script by:

    python2 prepare_htk_alignments_vctk.py ${your_vctk_data_path}

After running the script, you will see alignment files in `lab` directory as
follows:

    > tree ~/data/VCTK-Corpus/ -d -L

    /home/ryuichi/data/VCTK-Corpus/
    ├── lab
    ├── txt
    └── wav48
"""
import argparse
import logging
import multiprocessing
import os
import sys
from tqdm import tqdm
import json
from os.path import join, basename, dirname, exists
import numpy as np

import gentle
import librosa
from nnmnkwii.datasets import vctk


def on_progress(p):
    for k, v in p.items():
        logging.debug("%s: %s" % (k, v))


def write_hts_label(labels, lab_path):
    lab = ""
    for s, e, l in labels:
        s, e = float(s) * 1e7, float(e) * 1e7
        s, e = int(s), int(e)
        lab += "{} {} {}\n".format(s, e, l)
    print(lab)
    with open(lab_path, "w") as f:
        f.write(lab)


def json2hts(data):
    emit_bos = False
    emit_eos = False

    phone_start = 0
    phone_end = None
    labels = []

    for word in data["words"]:
        case = word["case"]
        if case != "success":
            raise RuntimeError("Alignment failed")
        start = float(word["start"])
        word_end = float(word["end"])

        if not emit_bos:
            labels.append((phone_start, start, "silB"))
            emit_bos = True

        phone_start = start
        phone_end = None
        for phone in word["phones"]:
            ph = str(phone["phone"][:-2])
            duration = float(phone["duration"])
            phone_end = phone_start + duration
            labels.append((phone_start, phone_end, ph))
            phone_start += duration
        assert np.allclose(phone_end, word_end)
    if not emit_eos:
        labels.append((phone_start, phone_end, "silE"))
        emit_eos = True

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Do force alignment for VCTK and save HTK-style alignments')
    parser.add_argument(
        '--nthreads', default=multiprocessing.cpu_count(), type=int,
        help='number of alignment threads')
    parser.add_argument(
        '--conservative', dest='conservative', action='store_true',
        help='conservative alignment')
    parser.set_defaults(conservative=False)
    parser.add_argument(
        '--disfluency', dest='disfluency', action='store_true',
        help='include disfluencies (uh, um) in alignment')
    parser.set_defaults(disfluency=False)
    parser.add_argument(
        '--log', default="INFO",
        help='the log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)')
    parser.add_argument('data_root', type=str, help='Data root')

    args = parser.parse_args()

    log_level = args.log.upper()
    logging.getLogger().setLevel(log_level)
    disfluencies = set(['uh', 'um'])

    data_root = args.data_root

    # Do for all speakers
    speakers = vctk.available_speakers

    # Collect all transcripts/wav files
    td = vctk.TranscriptionDataSource(data_root, speakers=speakers)
    transcriptions = td.collect_files()
    wav_paths = vctk.WavFileDataSource(
        data_root, speakers=speakers).collect_files()

    # Save dir
    save_dir = join(data_root, "lab")
    if not exists(save_dir):
        os.makedirs(save_dir)

    resources = gentle.Resources()

    for idx in tqdm(range(len(wav_paths))):
        transcript = transcriptions[idx]
        audiofile = wav_paths[idx]
        lab_path = audiofile.replace("wav48/", "lab/").replace(".wav", ".lab")
        print(transcript)
        print(audiofile)
        print(lab_path)
        lab_dir = dirname(lab_path)
        if not exists(lab_dir):
            os.makedirs(lab_dir)

        logging.info("converting audio to 8K sampled wav")
        with gentle.resampled(audiofile) as wavfile:
            logging.info("starting alignment")
            aligner = gentle.ForcedAligner(resources, transcript,
                                           nthreads=args.nthreads,
                                           disfluency=args.disfluency,
                                           conservative=args.conservative,
                                           disfluencies=disfluencies)
            result = aligner.transcribe(
                wavfile, progress_cb=on_progress, logging=logging)

            # convert to htk format
            a = json.loads(result.to_json())
            try:
                labels = json2hts(a)
            except RuntimeError as e:
                from warnings import warn
                warn(str(e))
                continue

            # Insert end time
            x, sr = librosa.load(wavfile, sr=8000)
            endtime = float(len(x)) / sr
            labels[-1] = (labels[-1][0], endtime, labels[-1][-1])

            # write to file
            write_hts_label(labels, lab_path)
