from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from SoundDataset import SoundDataset


class SoundDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds = SoundDataset(self.cfg, split="train")
        self.val_ds = SoundDataset(self.cfg, split="val")
        self.test_ds = SoundDataset(self.cfg, split="test")

    def train_dataloader(self):
        num_workers = self.cfg.dataloader.num_workers
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            persistent_workers=num_workers > 0,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
        )

    def val_dataloader(self):
        num_workers = self.cfg.dataloader.num_workers
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
        )

    def test_dataloader(self):
        num_workers = self.cfg.dataloader.num_workers
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            persistent_workers=num_workers > 0,
            num_workers=num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
        )
