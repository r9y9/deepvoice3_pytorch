# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

from deepvoice3_pytorch import frontend
from nose.plugins.attrib import attr

eos = 1


def test_en():
    f = getattr(frontend, "en")
    seq = f.text_to_sequence("hello world.")
    assert seq[-1] == eos
    t = f.sequence_to_text(seq)
    assert t == "hello world.~"


@attr("local_only")
def test_ja():
    f = getattr(frontend, "jp")
    seq = f.text_to_sequence("こんにちわ")
    assert seq[-1] == eos
    t = f.sequence_to_text(seq)
    assert t[:-1] == "コンニチワ。"


@attr("local_only")
def test_en_lj():
    f = getattr(frontend, "en")
    from nnmnkwii.datasets import ljspeech
    from tqdm import trange
    import jaconv

    d = ljspeech.TranscriptionDataSource("/home/ryuichi/data/LJSpeech-1.0")
    texts = d.collect_files()

    for p in [0.0, 0.5, 1.0]:
        for idx in trange(len(texts)):
            text = texts[idx]
            seq = f.text_to_sequence(text, p=p)
            assert seq[-1] == eos
            t = f.sequence_to_text(seq)

            if idx < 10:
                print("""{0}: {1}\n{0}: {2}\n""".format(idx, text, t))


@attr("local_only")
def test_ja_jsut():
    f = getattr(frontend, "jp")
    from nnmnkwii.datasets import jsut
    from tqdm import trange
    import jaconv

    d = jsut.TranscriptionDataSource("/home/ryuichi/data/jsut_ver1.1/",
                                     subsets=jsut.available_subsets)
    texts = d.collect_files()

    for p in [0.0, 0.5, 1.0]:
        for idx in trange(len(texts)):
            text = texts[idx]
            seq = f.text_to_sequence(text, p=p)
            assert seq[-1] == eos
            t = f.sequence_to_text(seq)

            if idx < 10:
                print("""{0}: {1}\n{0}: {2}\n""".format(idx, text, t))
