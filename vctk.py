from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
from nnmnkwii.datasets import vctk
from nnmnkwii.io import hts
from hparams import hparams
from os.path import exists
import librosa


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    speakers = vctk.available_speakers

    td = vctk.TranscriptionDataSource(in_dir, speakers=speakers)
    transcriptions = td.collect_files()
    speaker_ids = td.labels
    wav_paths = vctk.WavFileDataSource(
        in_dir, speakers=speakers).collect_files()

    for index, (speaker_id, text, wav_path) in enumerate(
            zip(speaker_ids, transcriptions, wav_paths)):
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, index + 1, speaker_id, wav_path, text)))
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, speaker_id, wav_path, text):
    sr = hparams.sample_rate

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    lab_path = wav_path.replace("wav48/", "lab/").replace(".wav", ".lab")

    # Trim silence from hts labels if available
    if exists(lab_path):
        labels = hts.load(lab_path)
        assert labels[0][-1] == "silB"
        assert labels[-1][-1] == "silE"
        b = int(labels[0][1] * 1e-7 * sr)
        e = int(labels[-1][0] * 1e-7 * sr)
        wav = wav[b:e]
    else:
        wav, _ = librosa.effects.trim(wav, top_db=15)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'vctk-spec-%05d.npy' % index
    mel_filename = 'vctk-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)
