from torch import tensor, rand

from cnn.model import VGG

def test_size():
    vgg = VGG()

    input = rand((2, 3, 224, 224))

    assert vgg.forward(input).shape == (2,1000)
