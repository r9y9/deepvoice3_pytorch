# deepvoice3_pytorch

[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)

PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: 2000-Speaker Neural Text-to-Speech.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

Current progress and planned TO-DOs can be found at [#1](https://github.com/r9y9/deepvoice3_pytorch/issues/1).

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/) and [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) datasets
- Language-dependent frontend text processor for English and Japanese

Support for multi-speaker models is planned but not completed yet.

## Audio samples

- [DeepVoice3] Samples from the model trained on LJ Speech Dataset: https://www.dropbox.com/sh/uq4tsfptxt0y17l/AADBL4LsPJRP2PjAAJRSH5eta?dl=0
- [Nyanko] Samples from the model trained on LJ Speech Dataset: https://www.dropbox.com/sh/q9xfgscgh3k5lqa/AACPgWCprBfNgjRravscdDYCa?dl=0

## Pretrained models

 | URL | Model      | Data     | Hyper paramters                                  | Git commit | Steps  |
 |-----|------------|----------|--------------------------------------------------|----------------------|--------|
 | [link](https://www.dropbox.com/s/4r207fq6s8gt2sl/20171213_deepvoice3_checkpoint_step00021000.pth?dl=0) | DeepVoice3 | LJSpeech | `--hparams="builder=deepvoice3,use_preset=True"` | [4357976](https://github.com/r9y9/deepvoice3_pytorch/tree/43579764f35de6b8bac2b18b52a06e4e11b705b2)| 210000 |
 |  [link](https://www.dropbox.com/s/j8ywsvs3kny0s0x/20171129_nyanko_checkpoint_step000585000.pth?dl=0)   | Nyanko     | LJSpeech | `--hparams="builder=nyanko,use_preset=True"`     | [ba59dc7](https://github.com/r9y9/deepvoice3_pytorch/tree/ba59dc75374ca3189281f6028201c15066830116) | 585000 |

See the `Synthesize from a checkpoint` section in the README for how to generate speech samples. Please make sure that you are on the specific git commit noted above.

## Notes on hyper parameters

- Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.
- `builder` specifies which model you want to use. `deepvoice3` [1] and `nyanko` [2] are surpprted.
- `presets` represents hyper parameters known to work well for LJSpeech dataset from my experiments. Before you try to find your best parameters, I would recommend you to try those presets by setting `use_preset=True`. E.g,
```
python train.py --data-root=./data/ljspeech --checkpoint-dir=checkpoints_deepvoice3 \
    --hparams="use_preset=True,builder=deepvoice3" \
    --log-event-path=log/deepvoice3_preset
```
or
```
python train.py --data-root=./data/ljspeech --checkpoint-dir=checkpoints_nyanko \
    --hparams="use_preset=True,builder=nyanko" \
    --log-event-path=log/nyanko_preset
```
- Hyper parameters described in DeepVoice3 paper for single speaker didn't work for LJSpeech dataset, so I changed a few things. Add dilated convolution, more channels, more layers and add guided loss, etc. See code for details.


## Requirements

- Python 3
- PyTorch >= v0.3
- TensorFlow >= v1.3
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (master)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.9
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)

## Installation

Please install packages listed above first, and then

```
git clone https://github.com/r9y9/deepvoice3_pytorch
pip install -e ".[train]"
```

If you want Japanese text processing frontend, install additional dependencies by:

```
pip install -e ".[jp]"
```

## Getting started

### 0. Download dataset

- LJSpeech (en): https://keithito.com/LJ-Speech-Dataset/
- JSUT (jp): https://sites.google.com/site/shinnosuketakamichi/publication/jsut

### 1. Preprocessing

Preprocessing can be done by `preprocess.py`. Usage is:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir}
```

Supported `${dataset_name}`s for now are `ljspeech` and `jsut`. Suppose you will want to preprocess LJSpeech dataset and have it in `~/data/LJSpeech-1.0`, then you can preprocess data by:

```
python preprocess.py ljspeech ~/data/LJSpeech-1.0/ ./data/ljspeech
```

When this is done, you will see extracted features (mel-spectrograms and linear spectrograms) in `./data/ljspeech`.

### 2. Training

Basic usage of `train.py` is:

```
python train.py --data-root=${data-root} --hparams="parameters you want to override"
```

Suppose you will want to build a DeepVoice3-style model using LJSpeech dataset with default hyper parameters, then you can train your model by:

```
python train.py --data-root=./data/ljspeech/ --hparams="use_preset=True,builder=deepvoice3"
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 5000 steps by default.

If you are building a Japaneses TTS model, then for example,

```
python train.py --data-root=./data/jsut --hparams="frontend=jp" --hparams="use_preset=True,builder=deepvoice3"
```

`frontend=jp` tell the training script to use Japanese text processing frontend. Default is `en` and uses English text processing frontend.

Note that there are many hyper parameters and design choices. Some are configurable by `hparams.py` and some are hardcoded in the source (e.g., dilation factor for each convolution layer). If you find better hyper parameters, please let me know!


### 4. Moniter with Tensorboard

Logs are dumped in `./log` directory by default. You can monitor logs by tensorboard:

```
tensorboard --logdir=log
```

### 5. Synthesize from a checkpoint

Given a list of text, `synthesis.py` synthesize audio signals from trained model. Usage is:

```
python synthesis.py ${checkpoint_path} ${text_list.txt} ${output_dir}
```

Example test_list.txt:

```
Generative adversarial network or variational auto-encoder.
Once upon a time there was a dear little girl who was loved by every one who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child.
A text-to-speech synthesis system typically consists of multiple stages, such as a text analysis frontend, an acoustic model and an audio synthesis module.
```

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/keithito/tacotron
- https://github.com/facebookresearch/fairseq-py
