import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from cnn.model import VGG
from cnn.data import read_ds, Sample

DEVICE = torch.device("cuda")

class DS(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _to_array(self, sample: Sample) -> torch.Tensor:
        img = sample.read()
        return torch.tensor(np.array(img.resize((224, 224))).transpose(2, 0, 1))

    def __getitem__(self, ix: int):
        return {
            "image": self._to_array(self.samples[ix]).to(DEVICE),
            "label": torch.tensor(self.samples[ix].label).to(DEVICE)
        }

def main():

    vgg = VGG().to(DEVICE)

    train_set = DS([*read_ds("train")])
    train_loader = DataLoader(train_set, 2)

    for batch in train_loader:

        vgg.forward(batch["image"])

        break


if __name__ == "__main__":
    main()
