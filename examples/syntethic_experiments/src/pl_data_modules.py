from typing import Any, List, Optional, Union

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from examples.syntethic_experiments.src.data_generation import DataGenerator


class BasePLDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig) -> None:
        super().__init__()
        self.conf = conf
        data_conf = self.conf.data
        self.dg = DataGenerator(
            num_classes=data_conf.num_classes,
            num_features=data_conf.num_features,
            num_samples=data_conf.num_samples,
            num_annotators=data_conf.num_annotators,
            aggregation_type=data_conf.aggregation_type,
            leak_distribution=data_conf.leak_distribution,
            equal_distribution=data_conf.equal_distribution,
            noise_matrix_type=data_conf.noise_matrix_type,
        )
        print("AGGREGATION TYPE:", data_conf.aggregation_type)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        data = self.dg.get_data("train", self.conf.data.min_t_diag_value)
        return DataLoader(
            data,
            shuffle=True,
            batch_size=self.conf.data.batch_size,
            num_workers=self.conf.data.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        data = self.dg.get_data("dev", self.conf.data.min_t_diag_value)
        return DataLoader(
            data,
            shuffle=False,
            batch_size=self.conf.data.batch_size,
            num_workers=self.conf.data.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        data = self.dg.get_data("test", None)
        return DataLoader(
            data,
            shuffle=False,
            batch_size=self.conf.data.batch_size,
            num_workers=self.conf.data.num_workers,
        )
