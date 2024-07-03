import torch
from torch.utils.data import Dataset, random_split as torch_random_split
import torch.types
import typing as t


_T = t.TypeVar("_T")


class DummyDataset(Dataset[int]):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return index


def random_split(src: t.Iterable[_T], ratio: t.Sequence[int | float]) -> list[list[_T]]:
    src_list = list(src)
    subsets = torch_random_split(
        DummyDataset(len(src_list)),
        ratio,
        generator=torch.Generator().manual_seed(42),
    )
    return [[src_list[i] for i in split.indices] for split in subsets]


def pad(x, max_len: int, pad_value: torch.types.Number = 0, dim: int = 0):
    # check whether x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    pad_shape = list(x.shape)
    pad_shape[dim] = max_len - x.shape[dim]
    padding = torch.full(pad_shape, pad_value, dtype=x.dtype)
    return torch.concat([x, padding], dim)


def pad_batch(
    x: list[torch.Tensor],
    max_len: int | None = None,
    pad_value: torch.types.Number = 0,
    dim: int = 0,
):
    if max_len is None:
        max_len = max(y.shape[dim] for y in x)
    return torch.stack([pad(y, max_len, pad_value, dim) for y in x])
