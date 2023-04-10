from copy import deepcopy
import torch
from torch import optim, nn
import lightning.pytorch as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from transformers import BertModel, BertTokenizerFast, DataCollatorWithPadding, BertTokenizer


class BaseModel(pl.LightningModule):

    def __init__(self, *args, learning_rate=5e-5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_metrics = {}
        self.val_metrics = {}
        self.learning_rate = learning_rate

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


class LinearEmbeddingTowerForClassification(BaseModel):
    """
    Embedding -- because we embed something with transformers
    Tower -- because we concat extra features to the embeddings
    Linear -- because the head is linear

    TODO: make embedder into a separate class; move preprocessor there.
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


class TokenClassification(BaseModel):
    def __init__(self, *args, num_labels=2, freeze=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer_pad = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels
        )       
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, num_labels)
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_labels = num_labels
        self.freeze = freeze

        self.train_metrics = {
            "accuracy": MulticlassAccuracy(
                num_classes=num_labels,
                average="none",
                ignore_index=-100
            ),
            "f1": MulticlassF1Score(
                num_classes=num_labels,
                average="none",
                ignore_index=-100
            )
        }
        self.val_metrics = deepcopy(self.train_metrics)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def tokenize_and_align_labels(self, tokenized, ner_tags, tokenizer):
        tokenized_inputs = tokenizer(tokenized, truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(ner_tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        max_l = max(map(lambda x: len(x), labels))
        for i in range(len(labels)):
            labels[i] = labels[i] + [-100] * (max_l - len(labels[i]))
        labels = torch.tensor(labels).to(self.device)
        # pad rest
        tokenized_inputs = self.collator(tokenized_inputs)
        for k, v in tokenized_inputs.items():
            tokenized_inputs[k] = v.to(self.device)
        return tokenized_inputs, labels
    
    def get_prediction(self, inputs):
        if self.freeze:
            with torch.no_grad():
                outputs = self.bert(**inputs)[0]
        else:
            outputs = self.bert(**inputs)[0]
        logits = self.head(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        batch_size = len(labels)
        inputs, labels = self.tokenize_and_align_labels(inputs, labels, tokenizer=self.tokenizer)
        # labels = inputs["labels"] 
        logits = self.get_prediction(inputs)

        logits = logits.view(-1, self.num_labels)
        labels = labels.flatten()
        
        loss = self.loss(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, batch_size=batch_size)
        self.update_metrics(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        batch_size = len(labels)
        inputs, labels = self.tokenize_and_align_labels(inputs, labels, tokenizer=self.tokenizer)
        # labels = inputs["labels"] 
        logits = self.get_prediction(inputs)

        logits = logits.view(-1, self.num_labels)
        labels = labels.flatten()
        
        loss = self.loss(logits, labels)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, batch_size=batch_size)
        self.update_metrics(logits, labels, partition="val")
        return loss