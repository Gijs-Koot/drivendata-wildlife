from collections.abc import Iterator
import csv
from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


CLASSES = (
    "ANTELOPE_DUIKER",
    "BIRD",
    "BLANK",
    "CIVET_GENET",
    "HOG",
    "LEOPARD",
    "MONKEY_PROSIMIAN",
    "RODENT",
)
DATA = Path(__file__).parent.parent / "data" / "dd"


@dataclass
class Sample:
    id: str
    path: Path
    site: str
    label: Optional[int]

    def read(self) -> Image.Image:
        return Image.open(self.path)


def read_ds(split: Literal["train", "test"]) -> Iterator[Sample]:
    """Read samples from the locally stored splits. Sample only has a label if
    it is from the "train" set."""


    labels = read_labels() if split == "train" else None

    with open(DATA / f"{split}_features.csv") as f:
        for record in csv.DictReader(f):
            yield Sample(
                id=record["id"],
                path=DATA / record["filepath"],
                site=record["site"],
                label=labels[record["id"]] if labels else None,
            )


def get_klass(r: dict) -> int:
    label = next(iter(l for l, v in r.items() if v == "1.0"))
    return CLASSES.index(label.upper())


def read_labels() -> dict[str, int]:
    with open(DATA / f"train_labels.csv") as f:
        return {
            r["id"]: get_klass(r) for r in csv.DictReader(f)
        }


def main():
    pass

if __name__ == "__main__":
    main()
