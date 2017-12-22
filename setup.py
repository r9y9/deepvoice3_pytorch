#!/usr/bin/env python

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.build_py
import os
import subprocess

version = '0.0.1'

# Adapted from https://github.com/pytorch/pytorch
cwd = os.path.dirname(os.path.abspath(__file__))
if os.getenv('TACOTRON_BUILD_VERSION'):
    version = os.getenv('TACOTRON_BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except subprocess.CalledProcessError:
        pass


class build_py(setuptools.command.build_py.build_py):

    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global version, cwd
        print('-- Building version ' + version)
        version_path = os.path.join(cwd, 'deepvoice3_pytorch', 'version.py')
        with open(version_path, 'w') as f:
            f.write("__version__ = '{}'\n".format(version))


class develop(setuptools.command.develop.develop):

    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)


setup(name='deepvoice3_pytorch',
      version=version,
      description='PyTorch implementation of Tacotron speech synthesis model.',
      packages=find_packages(),
      cmdclass={
          'build_py': build_py,
          'develop': develop,
      },
      install_requires=[
          "numpy",
          "scipy",
          "unidecode",
          "inflect",
          "librosa",
          "numba",
          "lws <= 1.0",
          "nltk",
      ],
      extras_require={
          "train": [
              "docopt",
              "tqdm",
              "tensorboardX",
              "nnmnkwii >= 0.0.11",
          ],
          "test": [
              "nose",
          ],
          "jp": [
              "jaconv",
              "mecab-python3",
          ],
      })
