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

SAMPLE_RATE = 128
 
class MyDataProvider(DataProvider[Path]):
    def __init__(self, data_root: str, split: list[float], preprocess: bool | None = False) -> None:
        npzs = sorted(Path(data_root).glob("**/*.npz"))
        sources = sorted({self.get_source(item) for item in npzs})
        sources_split = random_split(sources, split)
        self.paths = []
        mean_qualities = [[], [], []]
        if preprocess == False:
            for k in range(3):
                self.paths.append(np.load(f"dataset/{k}_path.npy"))
            return
        for k, src_split in enumerate(sources_split):
            path = []
            for i, src in enumerate(src_split):
                print(i, src, len(src_split))
                if not os.path.exists(f"dataset/bcg2psg/{src}"):
                    os.makedirs(f"dataset/bcg2psg/{src}")
                Preprocess(data_root, src, path, mean_qualities)
            self.paths.append(path)
            np.save(f"dataset/{k}_path.npy", np.array(path))
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(mean_qualities[0], bins=20, color='blue', alpha=0.7)
        plt.title('Mean BCG Quality')

        plt.subplot(1, 3, 2)
        plt.hist(mean_qualities[1], bins=20, color='red', alpha=0.7)
        plt.title('Mean ECG Quality')

        plt.subplot(1, 3, 3)
        plt.hist(mean_qualities[2], bins=20, color='green', alpha=0.7)
        plt.title('Mean RSP Quality')

        plt.tight_layout()
        plt.savefig("./data_quality.png")
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
    BCG: torch.Tensor
    ECG: torch.Tensor
    RSP: torch.Tensor


class MyDataset(Dataset[Sample]):
    def __init__(self, data_paths: list[Path]) -> None:
        self.data_paths = data_paths
        #print(self.data_paths)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Sample:
        bcg = np.load(f"{self.data_paths[idx]}_bcg.npy")
        ecg = np.load(f"{self.data_paths[idx]}_ecg.npy")
        rsp = np.load(f"{self.data_paths[idx]}_rsp.npy")
        return Sample(
            BCG=torch.from_numpy(bcg.copy()), 
            ECG=torch.from_numpy(ecg.copy()),
            RSP=torch.from_numpy(rsp.copy())
        )


class Batch(t.TypedDict):
    BCG: torch.Tensor
    ECG: torch.Tensor
    RSP: torch.Tensor


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
        bcg = [sample["BCG"] for sample in samples]
        ecg = [sample["ECG"] for sample in samples]
        rsp = [sample["RSP"] for sample in samples]
        return Batch(
            BCG=torch.stack(bcg, dim=0),
            ECG=torch.stack(ecg, dim=0),
            RSP=torch.stack(rsp, dim=0),
        )
if __name__ == "__main__":
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider", # "class": MyDataProvider,
            "data_root": "/local_data/datasets/exported_parallel_data_v2/",
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
