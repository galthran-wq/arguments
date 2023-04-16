from typing import Dict
import math
from copy import deepcopy
import torch
from torch import optim, nn
from torch.nn.init import kaiming_uniform
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from transformers import BertModel, BertTokenizerFast, DataCollatorWithPadding, BertTokenizer

import embedder
import head


class BaseModel(pl.LightningModule):

    def __init__(self, *args, learning_rate=5e-5, class_index_to_label=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_metrics = {}
        self.val_metrics = {}
        self._test_metrics = {}
        self.learning_rate = learning_rate

        if class_index_to_label is None:
            self.class_index_to_label = {}
        else:
            self.class_index_to_label = class_index_to_label
    
    def get_metrics_for_subset(self, subset):
        if subset == "train":
            return self.train_metrics
        elif subset == "val":
            return self.val_metrics
        else:
            return self.test_metrics

    def update_metrics(self, logits, y, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        pred = logits.argmax(dim=-1)
        for metric in metrics.values():
            metric.update(pred, y)

    def get_label_by_index(self, index):
        if self.class_index_to_label:
            return self.class_index_to_label[index]
        else:
            return index

    def get_metrics(self, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        results = {}
        for metric_name, metric in metrics.items():
            v = metric.compute()
            results.update({
                f"{subset}_{metric_name}_{self.get_label_by_index(i)}": v[i]
                for i in range(len(v))
            })
            results[f"{subset}_avg_{metric_name}"] = v.mean()
        return results

    def reset_metrics(self, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        for metric in metrics.values():
            metric.reset()

    def on_train_start(self) -> None:
        for metric in self.train_metrics.values():
            metric.to(self.device)

    def on_validation_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.to(self.device)

    def on_test_start(self) -> None:
        for metric in self.test_metrics.values():
            metric.to(self.device)
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.get_metrics())
        self.reset_metrics()
    
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.get_metrics(subset="val"))
        self.reset_metrics(subset="val")
    
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.get_metrics(subset="test"))
        self.reset_metrics(subset="test")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MainModel(BaseModel):
    def __init__(
        self, *args, 
        embedder: embedder.BaseEmbedder, 
        # aggregator: embedder.BaseEmbedder, 
        head: head.BaseHead, 
        learning_rate: float = 1e-4, 
        class_index_to_label=None, 
        pairs: bool = False,
        has_extra_features: bool = False,
        has_shared_features: bool = False,
        ignore_index: int = -100,
        **kwargs
    ) -> None:
        super().__init__(*args, learning_rate=learning_rate, class_index_to_label=class_index_to_label, **kwargs)
        self.embedder = embedder
        # self.aggregator = aggregator
        self.head = head

        self.ignore_index = ignore_index
        self.output_dim = head.output_dim
        self.pairs = pairs
        self.has_extra_features = has_extra_features
        self.has_shared_features = has_shared_features

        self.train_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=self.output_dim,
                average="none",
                ignore_index=ignore_index
            ),
            "f1": MulticlassF1Score(
                num_classes=self.output_dim,
                average="none",
                ignore_index=ignore_index
            )
        }
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def common_step(self, batch: Dict, batch_idx, subset="train"):
        to_embed1 = batch["sentence1"]
        to_embed2 = batch.get("sentence2", None)
        context = batch.get("context", None)
        extra_features1 = batch.get("extra_features1", None)
        extra_features2 = batch.get("extra_features2", None)
        shared_features = batch.get("shared_features", None)
        y = batch["labels"]

        embedded1 = self.embedder(
            to_embed=to_embed1, 
            context=context, 
            extra_features=extra_features1,
            device=self.device, 
        )
        if self.pairs:
            embedded2 = self.embedder(
                to_embed=to_embed2, 
                context=context, 
                device=self.device, 
                extra_features=extra_features2
            )
        # TODO: we should(?) allow for different strategies
        # aggregator(embedded1, embedded2, shared_features)
        to_concat = [embedded1]
        if self.pairs:
            to_concat.append(embedded2)
        if self.has_shared_features:
            to_concat.append(shared_features)
        embedded = torch.concat(to_concat, dim=-1)
        ###############
        logits = self.head(embedded).view(-1, self.output_dim)

        # some models classify tokens; so we loose (sequence_length) dimension
        logits = logits.view(-1, self.output_dim)
        y = y.flatten()

        loss = self.loss(logits, y)
        self.log(f"{subset}_loss", loss, batch_size=len(y))
        self.update_metrics(logits, y, subset=subset)
        return loss

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, subset="val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, subset="test")


class LinearEmbeddingTowerForClassification(BaseModel):
    """
    Embedding -- because we embed something with transformers
    Tower -- because we concat extra features to the embeddings
    Linear -- because the head is linear
    """
    def __init__(
            self, 
            embedder, 
            *args,
            extra_features_dim=0, 
            n_hidden_layers=3,
            output_dim=2,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.embedder = embedder
        self.embedding_dim = self.embedder.dim
        self.feature_dim = self.embedding_dim + extra_features_dim
        self.head = nn.Sequential(*([
            nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim // i, self.feature_dim // (i+1)),
                nn.ReLU()
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
                average="none",
            ),
            "f1": MulticlassF1Score(
                num_classes=output_dim,
                average="none",
            )
        }
        self.val_metrics = deepcopy(self.train_metrics)


    def get_prediction(self, to_embed, other_features, context=None):
        """todo: there might be no other features"""
        sentence_embedding = self.embedder(to_embed, device=self.device, context=context)
        logits = self.head(torch.concat([sentence_embedding, other_features.type(sentence_embedding.dtype)], dim=-1))
        return logits 

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
        context = batch["context"]
        other_features = batch["sentence_features"]
        y = batch["labels"]
        logits = self.get_prediction(to_embed, other_features, context=context)
        loss = self.loss(logits, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.update_metrics(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        to_embed = batch["sentence"]
        context = batch["context"]
        other_features = batch["sentence_features"]
        y = batch["labels"]
        logits = self.get_prediction(to_embed, other_features, context=context)
        loss = self.loss(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.update_metrics(logits, y, partition="val")
        return loss


class Simple(BaseModel):
    def __init__(
        self, 
        embedder, *args, 
        extra_features_dim=0, n_hidden_layers=3, 
        hidden_size=None, output_dim=2, shared_features_dim=0, 
        dropout=0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedder = embedder

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, output_dim),
        )
        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.BCEWithLogitsLoss()
        self.train_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=output_dim,
                average="none",
            ),
            "f1": MulticlassF1Score(
                num_classes=output_dim,
                average="none",
            )
        }
        self.val_metrics = deepcopy(self.train_metrics)
    
    def get_prediction(
        self, 
        to_embed1, other_features1,
        context,
        to_embed2, other_features2, shared_features
    ):
        hidden1 = self.embedder(to_embed1, device=self.device, context=context)
        hidden2 = self.embedder(to_embed2, device=self.device, context=context)
        logits = self.head(torch.concat([hidden1, hidden2], dim=-1))
        
        # logits = self.head(torch.concat([hidden1, hidden2, shared_features.type(hidden1.dtype)], dim=-1))
        return logits
        

    def training_step(self, batch, batch_idx):
        """
        TODO: refactor into separate classes to generalize for token classification.
        batch -- dictionary:
            - sentence: str
            - context: str
            - sentence_features: [batch_size, extra_features_dim]
            - labels: [batch_size]
        """
        to_embed1 = batch["sentence1"]
        to_embed2 = batch["sentence2"]
        context = batch["context"]
        extra_features1 = batch["extra_features1"]
        extra_features2 = batch["extra_features2"]
        shared_features = batch["shared_features"]
        y = batch["labels"]

        logits = self.get_prediction(
            to_embed1=to_embed1, 
            to_embed2=to_embed2,
            context=context,
            other_features1=extra_features1, 
            other_features2=extra_features2,
            shared_features=shared_features,
        )

        loss = self.loss(logits, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.update_metrics(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        to_embed1 = batch["sentence1"]
        to_embed2 = batch["sentence2"]
        context = batch["context"]
        extra_features1 = batch["extra_features1"]
        extra_features2 = batch["extra_features2"]
        shared_features = batch["shared_features"]
        y = batch["labels"]

        logits = self.get_prediction(
            to_embed1=to_embed1, 
            to_embed2=to_embed2,
            context=context,
            other_features1=extra_features1, 
            other_features2=extra_features2,
            shared_features=shared_features,
        )
        loss = self.loss(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.update_metrics(logits, y, partition="val")
        return loss


class LinearEmbeddingDoubleTowerForClassification(BaseModel):
    """
    We get its own tower for each of the two sentences.
    """
    def __init__(
        self, 
        embedder, *args, 
        extra_features_dim=0, n_hidden_layers=3, 
        hidden_size=None, output_dim=2, shared_features_dim=0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tower = LinearEmbeddingTowerForClassification(
            embedder, *args, 
            extra_features_dim=extra_features_dim, 
            n_hidden_layers=n_hidden_layers, 
            # Note
            output_dim=hidden_size, 
            **kwargs
        )

        # self.head = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(hidden_size * 2 + shared_features_dim, output_dim)
        # )
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.train_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=output_dim,
                average="none",
            ),
            "f1": MulticlassF1Score(
                num_classes=output_dim,
                average="none",
            )
        }
        self.dot_product = torch.nn.Parameter(torch.empty([hidden_size, hidden_size]))
        kaiming_uniform(self.dot_product, a=math.sqrt(5))
        
        self.val_metrics = deepcopy(self.train_metrics)
    
    def get_prediction(
        self, 
        to_embed1, other_features1,
        to_embed2, other_features2, shared_features
    ):
        hidden1 = self.tower.get_prediction(to_embed1, other_features1)
        hidden2 = self.tower.get_prediction(to_embed2, other_features2)
        logits = (hidden2 @ (self.dot_product @ hidden1.T)).diag()
        
        # logits = self.head(torch.concat([hidden1, hidden2, shared_features.type(hidden1.dtype)], dim=-1))
        return logits
        

    def training_step(self, batch, batch_idx):
        to_embed1 = batch["sentence1"]
        to_embed2 = batch["sentence2"]
        extra_features1 = batch["extra_features1"]
        extra_features2 = batch["extra_features2"]
        shared_features = batch["shared_features"]
        y = batch["labels"].type(torch.float16)

        logits = self.get_prediction(
            to_embed1=to_embed1, 
            to_embed2=to_embed2,
            other_features1=extra_features1, 
            other_features2=extra_features2,
            shared_features=shared_features
        )

        loss = self.loss(logits, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.update_metrics(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        to_embed1 = batch["sentence1"]
        to_embed2 = batch["sentence2"]
        extra_features1 = batch["extra_features1"]
        extra_features2 = batch["extra_features2"]
        shared_features = batch["shared_features"]
        y = batch["labels"].type(torch.float16)

        logits = self.get_prediction(
            to_embed1=to_embed1, 
            to_embed2=to_embed2,
            other_features1=extra_features1, 
            other_features2=extra_features2,
            shared_features=shared_features
        )
        loss = self.loss(logits, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        self.update_metrics(logits, y, partition="val")
        return loss
    
    def update_metrics(self, logits, y, partition="train"):
        pass
