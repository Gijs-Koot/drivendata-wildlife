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
        arr = np.array(img.resize((224, 224)).convert("RGB"))
        return torch.tensor(arr.transpose(2, 0, 1)) / 255

    def __getitem__(self, ix: int):
        return {
            "image": self._to_array(self.samples[ix]).to(DEVICE),
            "label": torch.tensor(self.samples[ix].label).to(DEVICE)
        }

def main():

    vgg = VGG().to(DEVICE)

    train_set = DS([*read_ds("train")])
    train_loader = DataLoader(train_set, 2)

    for param in vgg.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(vgg.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    for batch in train_loader:

        pred = vgg.forward(batch["image"])

        loss = loss_func(pred, batch["label"])

        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
