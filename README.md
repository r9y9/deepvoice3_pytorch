# Deepvoice3_pytorch

[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)

PyTorch implementation of convolutional networks-based text-to-speech synthesis models:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: 2000-Speaker Neural Text-to-Speech.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.

Audio sampels are available at https://r9y9.github.io/deepvoice3_pytorch/.

## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Multi-speaker and single speaker versions of DeepVoice3
- Audio samples and pre-trained models
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) datasets
- Language-dependent frontend text processor for English and Japanese

## Pretrained models

 | URL | Model      | Data     | Hyper paramters                                  | Git commit | Steps  |
 |-----|------------|----------|--------------------------------------------------|----------------------|--------|
 | [link](https://www.dropbox.com/s/cs6d070ommy2lmh/20171213_deepvoice3_checkpoint_step000210000.pth?dl=0) | DeepVoice3 | LJSpeech | `builder=deepvoice3,preset=deepvoice3_ljspeech` | [4357976](https://github.com/r9y9/deepvoice3_pytorch/tree/43579764f35de6b8bac2b18b52a06e4e11b705b2)| 21k ~ |
 |  [link](https://www.dropbox.com/s/1y8bt6bnggbzzlp/20171129_nyanko_checkpoint_step000585000.pth?dl=0)   | Nyanko     | LJSpeech | `builder=nyanko,preset=nyanko_ljspeech`     | [ba59dc7](https://github.com/r9y9/deepvoice3_pytorch/tree/ba59dc75374ca3189281f6028201c15066830116) | 58.5k |
  |  [link](https://www.dropbox.com/s/uzmtzgcedyu531k/20171222_deepvoice3_vctk108_checkpoint_step000300000.pth?dl=0)   | Multi-speaker DeepVoice3     | VCTK | `builder=deepvoice3_vctk,preset=deepvoice3_vctk`     | [0421749](https://github.com/r9y9/deepvoice3_pytorch/tree/0421749af908905d181f089f06956fddd0982d47) | 30k + 30k |

See "Synthesize from a checkpoint" section in the README for how to generate speech samples. Please make sure that you are on the specific git commit noted above.

## Notes on hyper parameters

- Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.
- `builder` specifies which model you want to use. `deepvoice3`, `deepvoice3_multispeaker` [1] and `nyanko` [2] are surpprted.
- `presets` represents hyper parameters known to work well for particular dataset/model from my experiments. Before you try to find your best parameters, I would recommend you to try those presets by setting `preset=${name}`. e.g., for LJSpeech, you can try either
```
python train.py --data-root=./data/ljspeech --checkpoint-dir=checkpoints_deepvoice3 \
    --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech" \
    --log-event-path=log/deepvoice3_preset
```
or
```
python train.py --data-root=./data/ljspeech --checkpoint-dir=checkpoints_nyanko \
    --hparams="builder=nyanko,preset=nyanko_ljspeech" \
    --log-event-path=log/nyanko_preset
```
- Hyper parameters described in DeepVoice3 paper for single speaker didn't work for LJSpeech dataset, so I changed a few things. Add dilated convolution, more channels, more layers and add guided attention loss, etc. See code for details. The changes are also applied for multi-speaker model.
- Multiple attention layers are hard to learn. Empirically, one or two (first and last) attention layers seems enough.
- With guided attention (see https://arxiv.org/abs/1710.08969), alignments get monotonic more quickly and reliably if we use multiple attention layers. With guided attention, I can confirm five attention layers get monotonic, though I cannot get speech quality improvements.
- Binary divergence (described in https://arxiv.org/abs/1710.08969) seems stabilizes training particularly for deep (> 10 layers) networks.
- Adam with step lr decay works. However, for deeper networks, I find Adam + noam's lr scheduler is more stable.

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
- VCTK (en): http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
- JSUT (jp): https://sites.google.com/site/shinnosuketakamichi/publication/jsut

### 1. Preprocessing

Preprocessing can be done by `preprocess.py`. Usage is:

```
python preprocess.py ${dataset_name} ${dataset_path} ${out_dir}
```

Supported `${dataset_name}`s for now are

- `ljspeech` (en, single speaker)
- `vctk` (en, multi-speaker)
- `jsut` (jp, single speaker)

Suppose you will want to preprocess LJSpeech dataset and have it in `~/data/LJSpeech-1.0`, then you can preprocess data by:

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
python train.py --data-root=./data/ljspeech/ --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"
```

Model checkpoints (.pth) and alignments (.png) are saved in `./checkpoints` directory per 5000 steps by default.

If you are building a Japaneses TTS model, then for example,

```
python train.py --data-root=./data/jsut --hparams="frontend=jp" --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"
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

## Advanced usage

### Multi-speaker model

Currently VCTK is the only supported dataset for building a multi-speaker model. Since some audio samples in VCTK have long silences that affect performance, it's recommended to do phoneme alignment and remove silences according to [vctk_preprocess](vctk_preprocess/).

Once you have phoneme alignment for each utterance, you can extract features by:

```
python preprocess.py vctk ${your_vctk_root_path} ./data/vctk
```

Now that you have data prepared, then you can train a multi-speaker version of DeepVoice3 by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --hparams="preset=deepvoice3_vctk,builder=deepvoice3_multispeaker" \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset
```

If you want to reuse learned embedding from other dataset, then you can do this instead by:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk \
   --hparams="preset=deepvoice3_vctk,builder=deepvoice3_multispeaker" \
   --log-event-path=log/deepvoice3_multispeaker_vctk_preset \
   --load-embedding=20171213_deepvoice3_checkpoint_step000210000.pth
```

This may improve training speed a bit.

### Speaker adaptation

If you have very limited data, then you can consider to try fine-turn pre-trained model. For example, using pre-trained model on LJSpeech, you can adapt it to data from VCTK speaker `p225` (30 mins) by the following command:

```
python train.py --data-root=./data/vctk --checkpoint-dir=checkpoints_vctk_adaptation \
    --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech" \
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
