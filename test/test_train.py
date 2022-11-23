from itertools import islice

import torch

from cnn.train import DS
from cnn.data import read_ds


def test_dtypes():
    ds = DS([*islice(read_ds("train"), 10)])

    assert len(ds.samples) == 10

    for sample in ds:
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (3, 224, 224)


def test_st():
    assert 1 + 1 == 2
