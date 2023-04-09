"""
TODO:
make runner into a separate class;
separate boilerplate code with defining characteristics -- model and ds
separate runner from cli
"""
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import ComponentIdentification, DictionaryCollator
from model import LinearEmbeddingTowerForClassification
from transformers import BertTokenizer, BertModel
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def cli_main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-s', '--seed', default=42, type=int)      
    parser.add_argument('--overfit-batch', action=argparse.BooleanOptionalAction)      
    parser.add_argument('--logs', default=False)      
    parser.add_argument('--patience', default=3, type=int)      
    # defult is no experiment
    parser.add_argument('--logdir', default="lightning_logs")      
    parser.add_argument('-name', '--experiment-name', default="")      
    parser.add_argument('-v', '--experiment-version', default=None)      
    # 
    parser.add_argument('--eval-period-epochs', default=1, type=int)      
    parser.add_argument('--max-train-epochs', default=100, type=int)      
    #
    parser.add_argument('--disable-checkpoints', action=argparse.BooleanOptionalAction)
    # model
    parser.add_argument('-lr', '--learning-rate', default=5e-5, type=float)      
    parser.add_argument('--freeze-embedder', action=argparse.BooleanOptionalAction)      
    return parser


if __name__ == "__main__":
    args = cli_main().parse_args()

    pl.seed_everything(args.seed, workers=True)

    ds = ComponentIdentification()
    generator = torch.Generator().manual_seed(args.seed)

    train, val = torch.utils.data.random_split(
        ds, [0.8, 0.2],
        generator=generator
    )
    if args.overfit_batch:
        train = torch.utils.data.Subset(train, range(10))

    train_dl = DataLoader(train, batch_size=8, collate_fn=DictionaryCollator())
    val_dl = DataLoader(val, batch_size=64, collate_fn=DictionaryCollator())

    bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = LinearEmbeddingTowerForClassification(
        preprocessor=tokenizer, 
        embedder=bert, 
        output_dim=3,
        learning_rate=args.learning_rate,
        freeze_embedder=args.freeze_embedder,
        extra_features_dim=ds.n_features
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,        # save the best model
        mode="min",
        every_n_epochs=1 if not args.disable_checkpoints else 0,
        save_last=True if not args.disable_checkpoints else False
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=args.patience, verbose=True, mode="min"
    )
    logger = TensorBoardLogger(
        args.logdir, 
        name=args.experiment_name, 
        version=args.experiment_version
    )
    logger.log_hyperparams(args)
    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=args.max_train_epochs, 
        deterministic=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=args.eval_period_epochs,
        logger=logger
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)