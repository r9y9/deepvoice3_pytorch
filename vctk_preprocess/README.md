# Preprocessing for VCTK

Wav files in VCTK contains lots of long silences, which affects training char-level seq2seq models. To deal with the problem, we will

- **Prepare phoneme alignments for all utterances** (code in the directory)
- Cut silences during preprocessing (code in the parent directory)

## Note

Code in the directory heavily relies on https://gist.github.com/kastnerkyle/cc0ac48d34860c5bb3f9112f4d9a0300 (which is hard copied in the repo). If you have any issues, please make sure that you can successfully run the script.

## Steps

1. Download VCTK: http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
2. Install HTK/speech_tools/festival/merlin and prepare `tts_env.sh`. If you don't have speech tools, you can install them by https://gist.github.com/kastnerkyle/001a58a58d090658ee5350cb6129f857. For the reference, `tts_env.sh` of mine is:
```
export ESTDIR=/home/ryuichi/Dropbox/sp/speech_tools/
export FESTDIR=/home/ryuichi/Dropbox/sp/festival/
export FESTVOXDIR=/home/ryuichi/Dropbox/sp/festvox/
export VCTKDIR=/home/ryuichi/data/VCTK-Corpus/
export HTKDIR=/usr/local/HTS-2.3/bin/
export SPTKDIR=/usr/local/bin/
export MERLINDIR=/home/ryuichi/Dropbox/sp/merlin_pr/
```
3. Run the script (takes ~24 hours)
```
python prepare_vctk_labels.py ${your_vctk_dir} ${dst_dir}
```
This will process all utterances of VCTK and copy HTK-style alignments to `${dst_dir}`.
It is recommended to copy alignments to the top of VCTK corpus. i.e.,
```
python prepare_vctk_labels.py ~/data/VCTK-Corpus ~/data/VCTK-Corpus/lab
```

After the above steps, you will get alignments as follows:

```                                                                                                              
tree ~/data/VCTK-Corpus/lab/ | head                                                                                                                      /home/ryuichi/data/VCTK-Corpus/lab/
├── p225
│   ├── p225_001.lab
│   ├── p225_002.lab
│   ├── p225_003.lab
│   ├── p225_004.lab
│   ├── p225_005.lab
│   ├── p225_006.lab
│   ├── p225_007.lab
│   ├── p225_008.lab
```

```
cat ~/data/VCTK-Corpus/lab/p225/p225_001.lab

         0     850000 pau
    850000    2850000 pau
   2850000    3600000 p
   3600000    3900000 l
   3900000    6000000 iy
   6000000    8450000 z
   8450000    8600000 k
   8600000   11300000 ao
  11300000   11450000 l
  11450000   12800000 s
  12800000   13099999 t
  13099999   15800000 eh
  15800000   16050000 l
  16050000   17600000 ax
  17600000   20400000 pau
```

## Using Gentle?

`prepare_htk_alignments_vctk.py` do the same things above using [Gentle](https://github.com/lowerquality/gentle), but turned out it seems not very good. Leaving code for future possibility if we can improve.
