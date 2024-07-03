import lightning as L
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict

from utils.create_object import create_instance

from .dataprovider import DataProvider


class BasicDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_provider: Dict[str, Any],
        dataset: Dict[str, Any],
        dataloader: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.data_provider_config = data_provider
        self._init_dataset_config(**dataset)
        self._init_dataloader_config(**dataloader)
        self.collate_fn = None

    def _init_dataset_config(self, train={}, eval={}, **common):
        self.train_dataset_config = {**common, **train}
        self.eval_dataset_config = {**common, **eval}

    def _init_dataloader_config(self, train={}, eval={}, **common):
        self.train_dataloader_config = {**common, **train}
        self.eval_dataloader_config = {**common, **eval}

    def setup(self, stage: str) -> None:
        data_provider = create_instance(DataProvider, self.data_provider_config)
        self.train_dataset = create_instance(
            Dataset, self.train_dataset_config, data_provider.train
        )
        self.valid_dataset = create_instance(
            Dataset, self.eval_dataset_config, data_provider.val
        )
        self.test_dataset = create_instance(
            Dataset, self.eval_dataset_config, data_provider.test
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            **self.train_dataloader_config,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            **self.eval_dataloader_config,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, **self.eval_dataloader_config, collate_fn=self.collate_fn
        )
