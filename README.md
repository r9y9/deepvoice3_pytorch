# deepvoice3_pytorch

[![Build Status](https://travis-ci.org/r9y9/deepvoice3_pytorch.svg?branch=master)](https://travis-ci.org/r9y9/deepvoice3_pytorch)

**NOTICE**: Work in progress! I haven't got any sucess yet.

PyTorch implementation of Deep Voice 3, a convolutional text-to-speech synthesis model described in https://arxiv.org/abs/1710.07654.


Current progress and planned to-dos can be found at https://github.com/r9y9/deepvoice3_pytorch/issues/1.

## Requirements

- PyTorch >= v0.2
- TensorFlow
- [fairseq](https://github.com/facebookresearch/fairseq-py)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) (master)

## Installation

```
git clone --recursive https://github.com/r9y9/deepvoice3_pytorch
pip install -e ".[train]"
```
