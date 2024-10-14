import sys
import os
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import typing as t
import numpy as np
from data.datamodule import BasicDataModule
from data.dataprovider import DataProvider
from data.utils import random_split
import matplotlib.pyplot as plt
from data.preprocess import Preprocess
from typing import Optional, List, Union
from scipy.stats import zscore


class MyDataProvider(DataProvider[Path]):
    def __init__(self, data_root: str, save_root: str, split: list[float], amount = 1.0, preprocess: Union[bool, None] = False) -> None:
        npzs = sorted(Path(data_root).glob("**/*.npz"))

        sources = sorted({self.get_source(item) for item in npzs})
        sources_split = random_split(sources, split)
        self.paths = []
        mean_qualities = [[], [], [], []]
        if preprocess == False:
            for k in range(3):
                self.paths.append(np.load(f"{save_root}/{k}_path.npy"))
            # use amount * len(self.paths[0]) to control the amount of training data
            # self.paths[0] = self.paths[0][:int(amount * len(self.paths[0]))]
            return
        for k, src_split in enumerate(sources_split):
            path = []
            for i, src in enumerate(src_split):
                print(i, src, len(src_split))
                if not os.path.exists(f"{save_root}/{src}"):
                    os.makedirs(f"{save_root}/{src}")
                Preprocess(data_root, save_root, src, path, mean_qualities)
            self.paths.append(path)
            np.save(f"{save_root}/{k}_path.npy", np.array(path))
        # plt.figure(figsize=(16, 4))

        # plt.subplot(1, 4, 1)
        # plt.hist(mean_qualities[0], bins=20, color='purple', alpha=0.7)
        # plt.title('Mean Trigger Quality')

        # plt.subplot(1, 4, 2)
        # plt.hist(mean_qualities[1], bins=20, color='blue', alpha=0.7)
        # plt.title('Mean BCG Quality')

        # plt.subplot(1, 4, 3)
        # plt.hist(mean_qualities[2], bins=20, color='red', alpha=0.7)
        # plt.title('Mean ECG Quality')

        # plt.subplot(1, 4, 4)
        # plt.hist(mean_qualities[3], bins=20, color='green', alpha=0.7)
        # plt.title('Mean RSP Quality')

        # plt.tight_layout()
        # plt.savefig("./data_quality.png")
    def get_source(self, item: Path) -> str:
        return item.parent.name

    def filter_by_sources(
        self, npzs: list[Path], source: list[str]
    ) -> list[Path]:
        return [item for item in npzs if self.get_source(item) in source]

    @property
    def train(self) -> list[Path]:
        return self.paths[0]

    @property
    def val(self) -> list[Path]:
        return self.paths[1]

    @property
    def test(self) -> list[Path]:
        return self.paths[2]


class Sample(t.TypedDict):
    input: torch.Tensor
    input_dec: torch.Tensor
    tar: torch.Tensor
    tar_dec: torch.Tensor

class MyDataset(Dataset[Sample]):
    def __init__(self, data_paths: list[Path]) -> None:
        self.data_paths = data_paths
        self.random_state = np.random.RandomState(0)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Sample:
        input = np.load(f"{self.data_paths[idx]}_bcg.npy")
        input_dec = np.load(f"{self.data_paths[idx]}_bcg_dec.npy")[0:1500] / 6000
        input_dec_real = input_dec.real
        input_dec_imag = input_dec.imag
        input_dec = np.stack([input_dec_real, input_dec_imag], axis = 0)
        tar =  np.load(f"{self.data_paths[idx]}_rsp.npy")
        tar_dec = np.load(f"{self.data_paths[idx]}_rsp_dec.npy") / 6000
        return Sample(
            input = torch.from_numpy(input.copy().astype(np.float32)),
            input_dec = torch.from_numpy(input_dec.copy().astype(np.float32)),
            tar = torch.from_numpy(tar.copy().astype(np.float32)),
            tar_dec = torch.from_numpy(tar_dec.copy().astype(np.complex64)),
        )


class Batch(t.TypedDict):
    input: torch.Tensor
    input_dec: torch.Tensor
    tar: torch.Tensor
    tar_dec: torch.Tensor
    length: torch.Tensor
    length_dec: torch.Tensor

class MyDataModule(BasicDataModule):
    def __init__(
        self,
        dataprovider: dict[str, t.Any],
        dataset: dict[str, t.Any],
        dataloader: dict[str, t.Any]
    ) -> None:
        super().__init__(dataprovider, dataset, dataloader)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, samples: list[Sample]) -> Batch:
        input = torch.stack([sample["input"] for sample in samples], dim = 0)
        input_dec = torch.stack([sample["input_dec"] for sample in samples], dim = 0)
        tar = torch.stack([sample["tar"] for sample in samples], dim = 0)
        tar_dec = torch.stack([sample["tar_dec"] for sample in samples], dim = 0)
        length = torch.tensor([sample["tar"].size(0) for sample in samples])
        length_dec = torch.tensor([sample["input_dec"].size(-1) for sample in samples])
        return Batch(
            input=input.type(torch.float32),
            input_dec=input_dec.type(torch.float32),
            tar=tar.type(torch.float32),
            tar_dec=tar_dec.type(torch.complex64),
            length=length,
            length_dec=length_dec,
        )
if __name__ == "__main__":
    save_root = "/nvme3/wenyule/bcg2psg/"
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider", # "class": MyDataProvider,
            "data_root": "/nvme4/preprocessed/wyl/exported_parallel_data_v2/",
            "save_root": save_root,
            "split": [0.90, 0.08, 0.02],
            "preprocess": True,
        },
        dataset={"class": "data.dataloader.MyDataset"},
        dataloader={
            "batch_size": 16,
            "num_workers": 4,
            "train": {"shuffle": True},
        },
    )
    dataloader.setup("fit")
    print(len(dataloader.train_dataloader()))
    print(len(dataloader.val_dataloader()))
    print(len(dataloader.test_dataloader()))
