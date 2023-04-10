from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from embedder import BertEmbedder, SBertEmbedder, ContextBertEmbedder
from dataset import ComponentClassification, DictionaryCollator, ComponentIdentification
from model import LinearEmbeddingTowerForClassification, TokenClassification


class BaseRunner:

    @dataclass
    class Arguments:
        """
        TODO: make Runner arguments explicit
        """
        pass

    def __init__(self, args: Arguments) -> None:
        self.args = args

    def get_collator(self):
        return DictionaryCollator()

    def get_dataset(self):
        raise NotImplemented
    
    def get_model(self, ds, embedder=None):
        raise NotImplemented
    
    def get_embedder(self):
        raise NotImplemented
    
    def partition_data(self, ds):
        generator = torch.Generator().manual_seed(self.args.seed)
        train, val = torch.utils.data.random_split(
            ds, [0.8, 0.2],
            generator=generator
        )
        if self.args.overfit_batch:
            train = torch.utils.data.Subset(train, range(10))
        return train, val
    
    def get_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=1,        # save the best model
            mode="min",
            every_n_epochs=1 if not self.args.disable_checkpoints else 0,
            save_last=True if not self.args.disable_checkpoints else False
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=self.args.patience, verbose=True, mode="min"
        )
        return [checkpoint_callback, early_stopping_callback]
    
    def get_logger(self):
        return TensorBoardLogger(
            self.args.logdir, 
            name=self.args.experiment_name, 
            version=self.args.experiment_version
        )


    def run(self):
        pl.seed_everything(self.args.seed, workers=True)
        ds = self.get_dataset()
        train, val = self.partition_data(ds)

        collator = self.get_collator()
        train_dl = DataLoader(train, batch_size=8, collate_fn=collator)
        val_dl = DataLoader(val, batch_size=64, collate_fn=collator)

        embedder = self.get_embedder()
        model = self.get_model(ds=ds, embedder=embedder)
        callbacks = self.get_callbacks()
        logger = self.get_logger()
        logger.log_hyperparams(self.args)

        trainer = pl.Trainer(
            limit_train_batches=100, 
            max_epochs=self.args.max_train_epochs, 
            deterministic=True,
            callbacks=callbacks,
            check_val_every_n_epoch=self.args.eval_period_epochs,
            logger=logger
        )
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


class CC_Runner(BaseRunner):
    def get_dataset(self):
        return ComponentClassification()

    def get_embedder(self):
        return BertEmbedder()

    def get_model(self, embedder, ds):
        """TODO: should somehow get rid of ds dependency"""
        return LinearEmbeddingTowerForClassification(
            embedder=embedder, 
            output_dim=3,
            learning_rate=self.args.learning_rate,
            extra_features_dim=ds.n_features
        )


class CC_RunnerSbert(BaseRunner):
    def get_dataset(self):
        return ComponentClassification()

    def get_embedder(self):
        return SBertEmbedder()

    def get_model(self, embedder, ds):
        """TODO: should somehow get rid of ds dependency"""
        return LinearEmbeddingTowerForClassification(
            embedder=embedder, 
            output_dim=3,
            learning_rate=self.args.learning_rate,
            extra_features_dim=ds.n_features
        )


class CC_RunnerContextBert(BaseRunner):
    def get_dataset(self):
        return ComponentClassification()

    def get_embedder(self):
        return ContextBertEmbedder()

    def get_model(self, embedder, ds):
        """TODO: should somehow get rid of ds dependency"""
        return LinearEmbeddingTowerForClassification(
            embedder=embedder, 
            output_dim=3,
            learning_rate=self.args.learning_rate,
            extra_features_dim=ds.n_features
        )


class CI_Runner(BaseRunner):
    def get_dataset(self):
        return ComponentIdentification()

    def get_embedder(self):
        return None
    
    def get_collator(self):
        def identity(batch):
            return [entry[0] for entry in batch], [entry[1] for entry in batch]
        return identity

    def get_model(self, ds, embedder=None):
        """TODO: should somehow get rid of ds dependency"""
        return TokenClassification(
            num_labels=ds.num_labels,
            learning_rate=self.args.learning_rate,
            freeze=self.args.freeze_embedder
        )