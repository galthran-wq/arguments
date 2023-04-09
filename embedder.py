import torch
from transformers import BertModel, BertTokenizer


class BaseEmbedder:
    def __call__(self, to_embed, context=None, device=None):
        raise NotImplemented


class BertEmbedder(BaseEmbedder):
    def __init__(self, bert_checkpoint="bert-base-uncased", freeze=True) -> None:
        self.bert = BertModel.from_pretrained(bert_checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)
        self.freeze = freeze
        self.dim = self.bert.config.hidden_size
    
    def __call__(self, to_embed, context=None, device=None):
        """
        Embed a sentence with the [CLS] token output.
        """
        inputs = self.tokenizer(to_embed, return_tensors="pt", padding=True)

        if device is not None:
            inputs = inputs.to(device)
            self.bert = self.bert.to(device)

        if self.freeze:
            with torch.no_grad():
                sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]
        else:
            sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]
        
        return sentence_embedding


class SBertEmbedder:
    pass


class ContextBertEmbedder:
    pass
