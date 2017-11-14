# deepvoice3_pytorch

[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)

PyTorch implementation of Deep Voice 3, a convolutional text-to-speech synthesis model described in https://arxiv.org/abs/1710.07654.


Current progress and planned TO-DOs can be found at [#1](https://github.com/r9y9/deepvoice3_pytorch/issues/1).

## Audio samples

- [WIP] Samples from the model trained on LJ Speech Dataset: https://www.dropbox.com/sh/uq4tsfptxt0y17l/AADBL4LsPJRP2PjAAJRSH5eta?dl=0

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/) and [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) datasets
- Language-dependent frontend text processor for English and Japanese

## Requirements

- Python 3
- PyTorch >= v0.2
- TensorFlow >= v1.3
- [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) (master)
- [fairseq](https://github.com/facebookresearch/fairseq-py) (master)
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

**Note**: Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.

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

`train.py` is the script for training models. Basic usage is:

```
python train.py --data-root=${data-root} --hparams="parameters you want to override"
```

Suppose you will want to build an English TTS model using LJSpeech dataset with default hyper parameters, then you can train your model by:

```
python train.py --data-root=./data/ljspeech/
```

Model checkpoints, alignments and predicted/target spectrograms are saved in `./checkpoints` directory per 5000 steps by default.

If you are building a Japaneses TTS model, then for example,

```
python train.py --data-root=./data/jsut --hparams="frontend=jp"
```

`frontend=jp` tell the training script to use Japanese text processing frontend. Default is `en` and uses English text processing frontend.

Note that there are many hyper parameters and design choices. Some are configurable by `hparams.py` and some are hardcoded in `deepvoice3_pytorch/deepvoice3.py` (e.g., dilation factor for each convolution layer). If you find better hyper parameters or model architectures, please let me know!


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
