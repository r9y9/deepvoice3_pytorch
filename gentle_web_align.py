# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 09:06:37 2018
Phoneme alignment and conversion in HTK-style label file using Web-served Gentle
This works on any type of english dataset.
Unlike prepare_htk_alignments_vctk.py, this is Python3 and Windows(with Docker) compatible.
Preliminary results show that gentle has better performance with noisy dataset
(e.g. movie extracted audioclips)
*This work was derived from vctk_preprocess/prepare_htk_alignments_vctk.py
@author: engiecat(github)

usage:
    gentle_web_align.py (-w wav_pattern) (-t text_pattern) [options]
    gentle_web_align.py (--nested-directories=<main_directory>) [options]

options:
    -w <wav_pattern> --wav_pattern=<wav_pattern> Pattern of wav files to be aligned
    -t <txt_pattern> --txt_pattern=<txt_pattern> Pattern of txt transcript files to be aligned (same name required)
    --nested-directories=<main_directory>        Process every wav/txt file in the subfolders of the given folder
    --server_addr=<server_addr>                  Server address that serves gentle. [default: localhost]
    --port=<port>                                Server port that serves gentle. [default: 8567]
    --max_unalign=<max_unalign>                  Maximum threshold for unalignment occurence (0.0 ~ 1.0) [default: 0.3] 
    --skip-already-done                          Skips if there are preexisting .lab file
    -h --help                                    show this help message and exit
"""

from docopt import docopt
from glob import glob
from tqdm import tqdm
import os.path
import requests
import numpy as np

def write_hts_label(labels, lab_path):
    lab = ""
    for s, e, l in labels:
        s, e = float(s) * 1e7, float(e) * 1e7
        s, e = int(s), int(e)
        lab += "{} {} {}\n".format(s, e, l)
    print(lab)
    with open(lab_path, "w", encoding='utf-8') as f:
        f.write(lab)


def json2hts(data):
    emit_bos = False
    emit_eos = False

    phone_start = 0
    phone_end = None
    labels = []
    failure_count = 0
    
    for word in data["words"]:
        case = word["case"]
        if case != "success":
            failure_count += 1 # instead of failing everything, 
            #raise RuntimeError("Alignment failed")
            continue
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
    unalign_ratio = float(failure_count) / len(data['words'])
    return unalign_ratio, labels


def gentle_request(wav_path,txt_path, server_addr, port, debug=False):
    print('\n')
    response = None
    wav_name = os.path.basename(wav_path)
    txt_name = os.path.basename(txt_path)
    if os.path.splitext(wav_name)[0] != os.path.splitext(txt_name)[0]:
        print(' [!] wav name and transcript name does not match - exiting...')
        return response
    with open(txt_path, 'r', encoding='utf-8-sig') as txt_file:
        print('Transcript - '+''.join(txt_file.readlines()))
    with open(wav_path,'rb') as wav_file, open(txt_path, 'rb') as txt_file:
        params = (('async','false'),)
        files={'audio':(wav_name,wav_file),
               'transcript':(txt_name,txt_file),
               }
        server_path = 'http://'+server_addr+':'+str(port)+'/transcriptions'
        response = requests.post(server_path, params=params,files=files)
        if response.status_code != 200:
            print(' [!] External server({}) returned bad response({})'.format(server_path, response.status_code))
    if debug:
        print('Response')
        print(response.json())
    return response

if __name__ == '__main__':
    arguments = docopt(__doc__)    
    server_addr = arguments['--server_addr']
    port = int(arguments['--port'])
    max_unalign  = float(arguments['--max_unalign'])
    if arguments['--nested-directories'] is None:
        wav_paths = sorted(glob(arguments['--wav_pattern']))
        txt_paths = sorted(glob(arguments['--txt_pattern']))    
    else:
        # if this is multi-foldered environment
        # (e.g. DATASET/speaker1/blahblah.wav)
        wav_paths=[]
        txt_paths=[]
        topdir = arguments['--nested-directories']
        subdirs = [f for f in os.listdir(topdir) if os.path.isdir(os.path.join(topdir, f))]
        for subdir in subdirs:
            wav_pattern_subdir = os.path.join(topdir, subdir, '*.wav')
            txt_pattern_subdir = os.path.join(topdir, subdir, '*.txt')
            wav_paths.extend(sorted(glob(wav_pattern_subdir)))
            txt_paths.extend(sorted(glob(txt_pattern_subdir)))
        
    t = tqdm(range(len(wav_paths)))
    for idx in t:
        try:
            t.set_description("Align via Gentle")
            wav_path = wav_paths[idx]
            txt_path = txt_paths[idx]
            lab_path = os.path.splitext(wav_path)[0]+'.lab'
            if os.path.exists(lab_path) and arguments['--skip-already-done']:
                print('[!] skipping because of pre-existing .lab file - {}'.format(lab_path))
                continue
            res=gentle_request(wav_path,txt_path, server_addr, port)
            unalign_ratio, lab = json2hts(res.json())
            print('[*] Unaligned Ratio - {}'.format(unalign_ratio))
            if unalign_ratio > max_unalign:
                print('[!] skipping this due to bad alignment')
                continue
            write_hts_label(lab, lab_path)
        except:
            # if sth happens, skip it
            import traceback
            tb = traceback.format_exc()
            print('[!] ERROR while processing {}'.format(wav_paths[idx]))
            print('[!] StackTrace - ')
            print(tb)

    