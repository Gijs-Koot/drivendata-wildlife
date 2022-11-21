from itertools import islice

from cnn import train


def test_read_ds():
    for r in islice(train.read_ds("train"), 100):
        assert r.label is not None

    for r in islice(train.read_ds("test"), 100):
        assert r.label is None

