"""
MiniTruth-8M dataset loader.

Loads token IDs from train.bin / val.bin and creates
input-target sequences for next-token prediction.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import config


class TokenDataset(Dataset):

    def __init__(
        self,
        split="train",
    ):
        if split not in ("train", "val"):
            raise ValueError(
                "split must be 'train' or 'val'"
            )

        file_path = Path(
            config.data.processed_dir
        ) / f"{split}.bin"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find {file_path}"
            )

        # Memory-map the file.
        #
        # This means we don't have to load the entire
        # dataset into RAM.
        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r",
        )

        self.block_size = (
            config.model.block_size
        )

    def __len__(self):

        return len(self.data) - self.block_size

    def __getitem__(self, index):

        # Example:
        #
        # data:
        # [10, 20, 30, 40, 50]
        #
        # input:
        # [10, 20, 30, 40]
        #
        # target:
        # [20, 30, 40, 50]

        chunk = self.data[
            index:index + self.block_size + 1
        ]

        chunk = torch.from_numpy(
            chunk.astype(np.int64)
        )

        x = chunk[:-1]
        y = chunk[1:]

        return x, y


def get_batch(
    dataset,
    batch_size,
    device="cpu",
):
    """
    Randomly sample a batch from the dataset.
    """

    max_index = len(dataset)

    indices = torch.randint(
        0,
        max_index,
        (batch_size,),
    )

    inputs = []
    targets = []

    for index in indices:

        x, y = dataset[
            index.item()
        ]

        inputs.append(x)
        targets.append(y)

    x = torch.stack(inputs)
    y = torch.stack(targets)

    return (
        x.to(device),
        y.to(device),
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing dataset")
    print("=" * 60)

    train_dataset = TokenDataset("train")
    val_dataset = TokenDataset("val")

    print(
        f"\nTraining tokens: "
        f"{len(train_dataset.data):,}"
    )

    print(
        f"Validation tokens: "
        f"{len(val_dataset.data):,}"
    )

    print(
        f"\nTraining examples: "
        f"{len(train_dataset):,}"
    )

    x, y = train_dataset[0]

    print(
        f"\nInput shape:  "
        f"{tuple(x.shape)}"
    )

    print(
        f"Target shape: "
        f"{tuple(y.shape)}"
    )

    print("\nFirst 20 input tokens:")
    print(x[:20].tolist())

    print("\nFirst 20 target tokens:")
    print(y[:20].tolist())

    print("\nDataset test successful.")