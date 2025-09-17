import contextlib
from itertools import cycle
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class CycledDataLoader:
    def __init__(
        self,
        datasets: list[Dataset | list | torch.Tensor],
        batch_size: int,
        **data_loader_kwargs,
    ):
        # Obtain the sub-batch size using integer division
        self.sub_batch_size = batch_size // len(datasets)

        # Store potential kwargs for the DataLoader class
        self._data_loader_kwargs = data_loader_kwargs

        # Separate the longest dataset from all others
        self._largest, *self._others = sorted(datasets, key=len, reverse=True)

    def _dataloaders(self) -> list[DataLoader]:
        # Use itertools.cycle to cycle the smaller datasets, but not the longest
        return [
            DataLoader(self._largest, self.sub_batch_size, **self._data_loader_kwargs)
        ] + [
            cycle(DataLoader(d, self.sub_batch_size, **self._data_loader_kwargs))
            for d in self._others
        ]

    @property
    def __len__(self) -> int:
        # The __len__ of this wrapper is just the number of sub-batches in the longest DataLoader
        return len(
            DataLoader(self._largest, self.sub_batch_size, **self._data_loader_kwargs)
        )

    def __iter__(self) -> Iterator[tuple[int, torch.Tensor]]:
        for batch_idx, sub_batches in enumerate(zip(*self._dataloaders())):
            yield batch_idx, torch.concat(sub_batches)


class ChainedDataLoader:
    def __init__(self, dataloaders: list[DataLoader]):
        """Create a DataLoader wrapper from a list of DataLoaders that iterates through all items of each dataloader consecutively."""
        self.dataloaders = dataloaders

    @property
    def __len__(self):
        return sum(dl.__len__() for dl in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader.__iter__()


class InterleavedDataLoader(ChainedDataLoader):
    """Create a DataLoader wrapper from a list of DataLoaders that yields items from the dataloaders."""

    def __iter__(self):
        dataloaders = [dataloader.__iter__() for dataloader in self.dataloaders]
        while dataloaders:
            dataloader = dataloaders.pop(0)
            with contextlib.suppress(StopIteration):
                yield next(dataloader)
                dataloaders.append(dataloader)


def example_cycle(drop_last=False):
    """
    Args:
        drop_last: If set to True, will drop the last batch of the LARGEST dataset if its size is not divisible by
            the sub_batch_size. If set False, the last batch may be unbalanced, if the size of the largest dataset is not
            divisible by the sub_batch_size!
    """
    data_1 = torch.randn((100, 128))
    data_2 = torch.randn((150, 128))
    data_3 = torch.randn((200, 128))

    batch_size = 45  # corresponds to sub_batch_size of 15 with 3 datasets
    cdl = CycledDataLoader([data_1, data_2, data_3], batch_size, drop_last=drop_last)
    for batch_idx, batch in cdl:
        print(batch_idx, batch.shape)


def example_break(drop_last=False):
    """
    Args:
        drop_last: If set to True, will drop the last batch of the SMALLEST dataset if its size is not divisible by
            the sub_batch_size. If set False, the last batch may be unbalanced, if the size of the smallest dataset is not
            divisible by the sub_batch_size!
    """
    data_1 = torch.randn((100, 128))
    data_2 = torch.randn((150, 128))
    data_3 = torch.randn((200, 128))

    sub_batch_size = 15  # corresponds to batch_size of 45 with 3 datasets
    dataloaders = [
        DataLoader(dataset, sub_batch_size, drop_last=drop_last)
        for dataset in (data_1, data_2, data_3)
    ]
    for batch_idx, sub_batches in enumerate(zip(*dataloaders)):
        batch = torch.concat(sub_batches)
        print(batch_idx, list(map(len, sub_batches)), batch.shape)


if __name__ == "__main__":
    print("example_break(True)")
    example_break(True)

    print()
    print("example_break(False)")
    example_break(False)

    print()
    print("example_cycle(False)")
    example_cycle(False)
