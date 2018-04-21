# Deepvoice3_pytorch

[![PyPI](https://img.shields.io/pypi/v/deepvoice3_pytorch.svg)](https://pypi.python.org/pypi/deepvoice3_pytorch)
[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)

PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: 2000-Speaker Neural Text-to-Speech.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

Audio samples are available at https://r9y9.github.io/deepvoice3_pytorch/.

## Online TTS demo

A notebook supposed to be executed on https://colab.research.google.com is available:

- [DeepVoice3: Multi-speaker text-to-speech demo](https://colab.research.google.com/github/r9y9/Colaboratory/blob/master/DeepVoice3_multi_speaker_TTS_en_demo.ipynb)

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Multi-speaker and single speaker versions of DeepVoice3
- Audio samples and pre-trained models
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) datasets
- Language-dependent frontend text processor for English and Japanese

### Samples

- [Ja Step000380000 Predicted](https://soundcloud.com/user-623907374/ja-step000380000-predicted)
- [Ja Step000370000 Predicted](https://soundcloud.com/user-623907374/ja-step000370000-predicted)
- [Ko_single Step000410000 Predicted](https://soundcloud.com/user-623907374/ko-step000410000-predicted)
- [Ko_single Step000400000 Predicted](https://soundcloud.com/user-623907374/ko-step000400000-predicted)
- [Ko_multi Step001680000 Predicted](https://soundcloud.com/user-623907374/step001680000-predicted)
- [Ko_multi Step001700000 Predicted](https://soundcloud.com/user-623907374/step001700000-predicted)

## Pretrained models

**NOTE**: pretrained models are not compatible to master. To be updated soon.

 | URL | Model      | Data     | Hyper paramters                                  | Git commit | Steps  |
 |-----|------------|----------|--------------------------------------------------|----------------------|--------|
 | [link](https://www.dropbox.com/s/cs6d070ommy2lmh/20171213_deepvoice3_checkpoint_step000210000.pth?dl=0) | DeepVoice3 | LJSpeech | `builder=deepvoice3,preset=deepvoice3_ljspeech` | [4357976](https://github.com/r9y9/deepvoice3_pytorch/tree/43579764f35de6b8bac2b18b52a06e4e11b705b2)| 210k ~ |
 |  [link](https://www.dropbox.com/s/1y8bt6bnggbzzlp/20171129_nyanko_checkpoint_step000585000.pth?dl=0)   | Nyanko     | LJSpeech | `builder=nyanko,preset=nyanko_ljspeech`     | [ba59dc7](https://github.com/r9y9/deepvoice3_pytorch/tree/ba59dc75374ca3189281f6028201c15066830116) | 585k |
  |  [link](https://www.dropbox.com/s/uzmtzgcedyu531k/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth?dl=0)   | Multi-speaker DeepVoice3     | VCTK | `builder=deepvoice3_multispeaker,preset=deepvoice3_vctk`     | [0421749](https://github.com/r9y9/deepvoice3_pytorch/tree/0421749af908905d181f089f06956fddd0982d47) | 300k + 300k |

See "Synthesize from a checkpoint" section in the README for how to generate speech samples. Please make sure that you are on the specific git commit noted above.

## Notes on hyper parameters

- Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.
- `builder` specifies which model you want to use. `deepvoice3`, `deepvoice3_multispeaker` [1] and `nyanko` [2] are surpprted.
- Hyper parameters described in DeepVoice3 paper for single speaker didn't work for LJSpeech dataset, so I changed a few things. Add dilated convolution, more channels, more layers and add guided attention loss, etc. See code for details. The changes are also applied for multi-speaker model.
- Multiple attention layers are hard to learn. Empirically, one or two (first and last) attention layers seems enough.
- With guided attention (see https://arxiv.org/abs/1710.08969), alignments get monotonic more quickly and reliably if we use multiple attention layers. With guided attention, I can confirm five attention layers get monotonic, though I cannot get speech quality improvements.
- Binary divergence (described in https://arxiv.org/abs/1710.08969) seems stabilizes training particularly for deep (> 10 layers) networks.
- Adam with step lr decay works. However, for deeper networks, I find Adam + noam's lr scheduler is more stable.

## Requirements

- Python 3
- CUDA >= 8.0
- TensorFlow >= v1.3
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.11
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)

## Installation

Please install packages listed above first, and then

```
git clone https://github.com/r9y9/deepvoice3_pytorch && cd deepvoice3_pytorch
pip install -e ".[train]"
```

## Getting started

### Preset parameters

There are many hyper parameters to be turned depends on what model and data you are working on. For typical datasets and models, parameters that known to work good (**preset**) are provided in the repository. See `presets` directory for details. Notice that

1. `preprocess.py`
2. `train.py`
3. `synthesis.py`

accepts `--preset=<json>` optional parameter, which specifies where to load preset parameters. If you are going to use preset parameters, then you must use same `--preset=<json>` throughout preprocessing, training and evaluation. e.g.,

```
python preprocess.py --preset=presets/deepvoice3_ljspeech.json ljspeech ~/data/LJSpeech-1.0
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech
```

instead of

```
python preprocess.py ljspeech ~/data/LJSpeech-1.0
# warning! this may use different hyper parameters used at preprocessing stage
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech
```

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- VCTK (en): http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
- JSUT (jp): https://sites.google.com/site/shinnosuketakamichi/publication/jsut
- NIKL (ko): http://www.korean.go.kr/front/board/boardStandardView.do?board_id=4&mn_id=17&b_seq=464

### 1. Preprocessing

Usage:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir} --preset=<json>
```

Supported `${dataset_name}`s are:

- `ljspeech` (en, single speaker)
- `vctk` (en, multi-speaker)
- `jsut` (jp, single speaker)
- `nikl_m` (ko, multi-speaker)
- `nikl_s` (ko, single speaker)

Assuming you use preset parameters known to work good for LJSpeech dataset / DeepVoice3 and have data in `~/data/LJSpeech-1.0`, then you can preprocess data by:

```
python preprocess.py --preset=presets/deepvoice3_ljspeech.json ljspeech ~/data/LJSpeech-1.0/ ./data/ljspeech
```

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `./data/ljspeech`.

### 2. Training

Usage:

```
python train.py --data-root=${data-root} --preset=<json> --hparams="parameters you may want to override"
```

Suppose you build a DeepVoice3-style model using LJSpeech dataset, then you can train your model by:

```
python train.py --preset=presets/deepvoice3_ljspeech.json --data-root=./data/ljspeech/
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 10000 steps by default.

#### NIKL

Pleae check [this](https://github.com/homink/deepvoice3_pytorch/blob/master/nikl_preprocess/README.md) in advance and follow the commands below.

```
python preprocess.py nikl_s ${your_nikl_root_path} data/nikl_s --preset=presets/deepvoice3_nikls.json

python train.py --data-root=./data/nikl_s --checkpoint-dir checkpoint_nikl_s --preset=presets/deepvoice3_nikls.json
```

### 4. Monitor with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 5. Synthesize from a checkpoint

Given a list of text, `synthesis.py` synthesize audio signals from trained model. Usage is:

```
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir} --preset=<json>
```

Example test_list.txt:

```
Generative adversarial network or variational auto-encoder.
Once upon a time there was a dear little girl who was loved by every one who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child.
A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module.
```

## Advanced usage

### Multi-speaker model

VCTK and NIKL are supported dataset for building a multi-speaker model.

#### VCTK
Since some audio samples in VCTK have long silences that affect performance, it's recommended to do phoneme alignment and remove silences according to [vctk_preprocess](vctk_preprocess/).

Once you have phoneme alignment for each utterance, you can extract features by:

```
python preprocess.py vctk ${your_vctk_root_path} ./data/vctk
```

Now that you have data prepared, then you can train a multi-speaker version of DeepVoice3 by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --preset=presets/deepvoice3_vctk.json \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset
```

If you want to reuse learned embedding from other dataset, then you can do this instead by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --preset=presets/deepvoice3_vctk.json \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset \
   --load-embedding=20171213_deepvoice3_checkpoint_step000210000.pth
```

This may improve training speed a bit.

#### NIKL

You will be able to obtain cleaned-up audio samples in ../nikl_preprocoess. Details are found in [here](https://github.com/homink/speech.ko).


Once NIKL corpus is ready to use from the preprocessing, you can extract features by:

```
python preprocess.py nikl_m ${your_nikl_root_path} data/nikl_m
```

Now that you have data prepared, then you can train a multi-speaker version of DeepVoice3 by:

```
python train.py --data-root=./data/nikl_m  --checkpoint-dir checkpoint_nikl_m \
   --preset=presets/deepvoice3_niklm.json
```

### Speaker adaptation

If you have very limited data, then you can consider to try fine-turn pre-trained model. For example, using pre-trained model on LJSpeech, you can adapt it to data from VCTK speaker `p225` (30 mins) by the following command:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk_adaptation \
    --preset=presets/deepvoice3_ljspeech.json \
    --log-event-path=log/deepvoice3_vctk_adaptation \
    --restore-parts="20171213_deepvoice3_checkpoint_step000210000.pth"
    --speaker-id=0
```

From my experience, it can get reasonable speech quality very quickly rather than training the model from scratch.

There are two important options used above:

- `--restore-parts=<N>`: It specifies where to load model parameters. The differences from the option `--checkpoint=<N>` are 1) `--restore-parts=<N>` ignores all invalid parameters, while `--checkpoint=<N>` doesn't. 2) `--restore-parts=<N>` tell trainer to start from 0-step, while `--checkpoint=<N>` tell trainer to continue from last step. `--checkpoint=<N>` should be ok if you are using exactly same model and continue to train, but it would be useful if you want to customize your model architecture and take advantages of pre-trained model.
- `--speaker-id=<N>`: It specifies what speaker of data is used for training. This should only be specified if you are using multi-speaker dataset. As for VCTK, speaker id is automatically assigned incrementally (0, 1, ..., 107) according to the `speaker_info.txt` in the dataset.

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/keithito/tacotron
- https://github.com/facebookresearch/fairseq-py
