from copy import deepcopy
import torch
from transformers import BertModel, BertTokenizer
from torch import optim, nn
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")


class LinearEmbeddingTowerForClassification(pl.LightningModule):
    """
    Embedding -- because we embed something with transformers
    Tower -- because we concat extra features to the embeddings
    Linear -- because the head is linear

    TODO: make embedder into a separate class; move preprocessor there.
    """
    def __init__(
            self, 
            embedder, 
            extra_features_dim=0, 
            n_hidden_layers=3,
            output_dim=2,
            learning_rate=5e-5,
        ):
        super().__init__()
        self.embedder = embedder
        self.embedding_dim = self.embedder.dim
        self.feature_dim = self.embedding_dim + extra_features_dim
        self.learning_rate = learning_rate
        self.head = nn.Sequential(*([
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim // i, self.feature_dim // (i+1))
            )
            for i in range(1, n_hidden_layers)
        ] + [
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim // n_hidden_layers, output_dim)
            )
        ]))
        self.loss = nn.CrossEntropyLoss()
        self.train_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=output_dim,
                average="none"
            ),
            "f1": MulticlassF1Score(
                num_classes=output_dim,
                average="none"
            )
        }
        self.val_metrics = deepcopy(self.train_metrics)


    def get_prediction(self, to_embed, other_features):
        sentence_embedding = self.embedder(to_embed, device=self.device)
        logits = self.head(torch.concat([sentence_embedding, other_features.type(sentence_embedding.dtype)], dim=-1))
        return logits 

    def update_metrics(self, logits, y, partition="train"):
        metrics = self.train_metrics if partition=="train" else self.val_metrics
        pred = logits.argmax(dim=-1)
        for metric in metrics.values():
            metric.update(pred, y)

    def get_metrics(self, partition="train"):
        metrics = self.train_metrics if partition=="train" else self.val_metrics
        results = {}
        for metric_name, metric in metrics.items():
            v = metric.compute()
            results.update({
                f"{partition}_{metric_name}_{i}": v[i]
                for i in range(len(v))
            })
            results[f"{partition}_avg_{metric_name}"] = v.mean()
        return results
    
    def reset_metrics(self, partition="train"):
        metrics = self.train_metrics if partition=="train" else self.val_metrics
        for metric in metrics.values():
            metric.reset()

    def training_step(self, batch, batch_idx):
        """
        TODO: refactor into separate classes to generalize for token classification.
        batch -- dictionary:
            - sentence: str
            - context: str
            - sentence_features: [batch_size, extra_features_dim]
            - labels: [batch_size]
        """
        to_embed = batch["sentence"]
        other_features = batch["sentence_features"]
        y = batch["labels"]
        logits = self.get_prediction(to_embed, other_features)
        loss = self.loss(logits, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.update_metrics(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        to_embed = batch["sentence"]
        other_features = batch["sentence_features"]
        y = batch["labels"]
        logits = self.get_prediction(to_embed, other_features)
        loss = self.loss(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.update_metrics(logits, y, partition="val")
        return loss
    
    def on_train_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.to(self.device)

    def on_validation_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.to(self.device)
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.get_metrics())
        self.reset_metrics()
    
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.get_metrics(partition="val"))
        self.reset_metrics(partition="val")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
