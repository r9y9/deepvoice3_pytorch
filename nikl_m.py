from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
import re

from hparams import hparams


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.

    # You will need to modify and format NIKL transcrption file will UTF-8 format
    # please check https://github.com/homink/deepspeech.pytorch.ko/blob/master/data/local/clean_corpus.sh

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    spk_id = {}
    with open(in_dir + '/speaker.mid', encoding='utf-8') as f:
        for i, line in enumerate(f):
            spk_id[line.rstrip()] = i

    index = 1
    with open(in_dir + '/metadata.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = parts[0]
            text = parts[1]
            uid = re.search(r'([a-z][a-z][0-9][0-9]_t)', wav_path)
            uid = uid.group(1).replace('_t', '')
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index + 1, spk_id[uid], wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'nikl-multi-spec-%05d.npy' % index
    mel_filename = 'nikl-multi-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
