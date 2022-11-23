from itertools import islice

import torch

from cnn.train import DS
from cnn.data import read_ds


def test_dtypes():
    ds = DS([*islice(read_ds("train"), 5)])
    for sample in ds:
        assert sample["image"].dtype == torch.FloatTensor
        assert sample["image"].shape == (3, 244, 244)
