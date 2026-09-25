import hydra, pytorch_lightning as pl
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import SoundDataModule
from lightning_module import RIRLightning


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    log_root = Path("results") / cfg.apt / cfg.exp_name
    logger = TensorBoardLogger(
        save_dir=str(log_root.parent), name=log_root.name, default_hp_metric=False
    )
    data = SoundDataModule(cfg)
    model = RIRLightning(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        log_every_n_steps=50,
        deterministic=True,
        enable_checkpointing=True,
        default_root_dir=str(log_root),
    )

    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
