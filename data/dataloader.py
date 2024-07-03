import sys
import os
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
import typing as t
from scipy.stats import zscore
import neurokit2 as nk
import numpy as np
from data.datamodule import BasicDataModule
from data.dataprovider import DataProvider
from data.utils import random_split
import matplotlib.pyplot as plt


SAMPLE_RATE = 128


class MyDataProvider(DataProvider[Path]):
    def __init__(self, data_root: str, split: list[float], preprocess: bool | None = False) -> None:
        npzs = sorted(Path(data_root).glob("**/*.npz"))
        sources = sorted({self.get_source(item) for item in npzs})
        sources_split = random_split(sources, split)
        self.paths = []
        if preprocess == False:
            for k in range(3):
                self.paths.append(np.load(f"dataset/{k}_path.npy"))
            return
        for k, src_split in enumerate(sources_split):
            path = []
            for i, src in enumerate(src_split):
                print(i, src)
                if os.path.exists(f"{data_root}/{src}/psg_quality_ecg_ECG II.npz") == False:
                    continue
                with np.load(f"{data_root}/{src}/bcg_feature_raw.npz") as f:
                    bcg_feature = f["raw"]
                    bcg_freq = f["fs"]
                with np.load(f"{data_root}/{src}/psg_feature_ecg_ECG II_raw.npz") as f:
                    ecg_feature = f["ecg"]
                    ecg_freq = f["fs"]

                with np.load(f"{data_root}/{src}/bcg_quality_bio_freq.npz") as f:
                    bcg_quality = f["quality"]
                    bcg_quality_freq = f["fs"]
                with np.load(f"{data_root}/{src}/psg_quality_ecg_ECG II.npz") as f:
                    ecg_quality = f["quality"]
                    ecg_quality_freq = f["fs"]
                for l in range(0, int(len(bcg_quality)//bcg_quality_freq) - 100, 1000):
                    r = l + 10
                    # bcg_q.append(np.mean(bcg_quality[int(l * bcg_quality_freq) : int(r * bcg_quality_freq)]))
                    if np.mean(bcg_quality[int(l * bcg_quality_freq) : int(r * bcg_quality_freq)]) < 0.6:
                        continue
                    # ecg_q.append(np.mean(ecg_quality[int(l * ecg_quality_freq) : int(r * ecg_quality_freq)]))
                    if np.mean(ecg_quality[int(l * ecg_quality_freq) : int(r * ecg_quality_freq)]) < 0.9:
                        continue
                    bcg = bcg_feature[l * bcg_freq : r * bcg_freq].astype(np.float32)
                    bcg = nk.signal_filter(bcg, sampling_rate=bcg_freq.item(), lowcut=3, highcut=11).astype(np.float32)
                    bcg = zscore(bcg)
                    ecg = ecg_feature[l * ecg_freq : r * ecg_freq].astype(np.float32)
                    ecg = nk.ecg_clean(ecg, sampling_rate = ecg_freq.item())
                    ecg = nk.signal_resample(ecg, desired_length = len(bcg))
                    ecg, _ = nk.ecg_invert(ecg, sampling_rate = ecg_freq.item())
                    ecg = nk.signal_filter(ecg, sampling_rate=ecg_freq.item(), lowcut=0, highcut=20).astype(np.float32)
                    ecg = zscore(ecg)
                    # save bcg and ecg
                    np.save(f"{data_root}/{src}/{l}_bcg.npy", bcg)
                    np.save(f"{data_root}/{src}/{l}_ecg.npy", ecg)
                    path.append(f"{data_root}/{src}/{l}")
            self.paths.append(path)
            np.save(f"dataset/{k}_path.npy", np.array(path))
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


class MyDataset(Dataset[Sample]):
    def __init__(self, data_paths: list[Path]) -> None:
        self.data_paths = data_paths
        #print(self.data_paths)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> Sample:
        bcg = np.load(f"{self.data_paths[idx]}_bcg.npy")
        ecg = np.load(f"{self.data_paths[idx]}_ecg.npy")
        return Sample(BCG=torch.from_numpy(bcg.copy()), ECG=torch.from_numpy(ecg.copy()))


class Batch(t.TypedDict):
    input: torch.Tensor
    target: torch.Tensor


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
        return Batch(input=torch.stack(bcg), target=torch.stack(ecg)) 
if __name__ == "__main__":
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider", # "class": MyDataProvider,
            "data_root": "./dataset/bcg2psg",
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
    for batch in dataloader.train_dataloader():
        bcg = batch["input"][0].numpy()
        ecg = batch["target"][0].numpy()
        plt.figure()
        plt.subplot(2, 1, 1) 
        plt.plot(bcg, label='BCG')
        plt.legend()
        plt.title('BCG Data')
        plt.subplot(2, 1, 2)  
        plt.plot(ecg, label='ECG')
        plt.legend()
        plt.title('ECG Data')
        plt.tight_layout()
        plt.savefig("data.png")
        break

