import os

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from examples.syntethic_experiments.src.pl_data_modules import BasePLDataModule
from examples.syntethic_experiments.src.pl_modules import BasePLModule
from examples.syntethic_experiments.src.utils import read_json, write_json
from examples.cifar10n_experiments.utils import seed_everything


def train(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    # main module declaration
    pl_module = BasePLModule(conf)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback
        )
        callbacks_store.append(model_checkpoint_callback)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    if conf.train.model_checkpoint_callback is not None:
        # retrieve the best model accordingly with model_checkpoint_callback
        checkpoint_path: str = model_checkpoint_callback.best_model_path
        pl_module.load_from_checkpoint(checkpoint_path)
    # module test
    results = trainer.test(pl_module, datamodule=pl_data_module)

    # We want to log also the best loss on the validation set for each method

    results[0]["dev_loss"] = model_checkpoint_callback.best_model_score.item()
    results[0]["wrong_predictions_percentage"] = 100 * round(
        (
            pl_module.total_errors["test"]
            / len(pl_data_module.dg.get_data("test", None))
        ),
        5,
    )
    json_path = hydra.utils.to_absolute_path(
        f"experiments/out/{conf.train.experiment_name}/{conf.train.seed}/"
    )
    os.makedirs(json_path, exist_ok=True)
    json_filename = (
        f"output_{conf.data.aggregation_type}_{conf.data.min_t_diag_value}.json"
    )
    try:
        dictionary_to_dump = read_json()
    except:
        dictionary_to_dump = {}
    dictionary_to_dump[conf.data.min_t_diag_value] = {
        conf.data.aggregation_type: results
    }
    write_json(json_path + f"/{conf.train.seed}" + json_filename, dictionary_to_dump)


@hydra.main(config_path="../conf", config_name="root", version_base="1.1")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
