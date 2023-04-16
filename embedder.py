from typing import Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from algs import get_subarray_index
from tqdm.auto import tqdm


class BaseEmbedder(nn.Module):
    def forward(self, to_embed, context=None, device=None, extra_features=None):
        raise NotImplemented


class BertClsEmbedder(BaseEmbedder):
    def __init__(self, bert_checkpoint="bert-base-uncased", freeze=True) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)
        self.freeze = freeze
        self.dim = self.bert.config.hidden_size
    
    def forward(self, to_embed, context=None, device=None, extra_features=None):
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
        
        if extra_features is not None:
            sentence_embedding = torch.concat([
                sentence_embedding, 
                (extra_features).type(sentence_embedding.dtype)
            ], dim=-1)

        return sentence_embedding


class BertEmbedder(BaseEmbedder):
    def __init__(
        self, 
        bert_checkpoint="bert-base-uncased",
        freeze=True,
        is_tokenized=True
    ) -> None:
        super().__init__()
        self.is_tokenized = is_tokenized
        self.bert = BertModel.from_pretrained(bert_checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)
        self.freeze = freeze
        self.dim = self.bert.config.hidden_size
    
    def forward(
        self, 
        to_embed: Union[Dict[str, torch.Tensor], str] , 
        context=None, 
        device=None, 
        extra_features=None
    ):
        """Get embeddings for each token in the input sentence.

        Args:
            to_embed (Union[Dict[str, torch.Tensor], str]): If the sentence is already tokenized, then these are the inputs to the BertModel. If not, it is a sentence
            context ([type], optional): [description]. Defaults to None.
            device ([type], optional): [description]. Defaults to None.
            extra_features ([type], optional): Does not support extra features.

        Returns:
            [type]: [description]
        """
        if self.is_tokenized:
            inputs = to_embed
        else:
            inputs = self.tokenizer(to_embed, return_tensors="pt", padding=True)

        if device is not None:
            inputs = inputs.to(device)
            self.bert = self.bert.to(device)

        if self.freeze:
            with torch.no_grad():
                sentence_embedding = self.bert(**inputs).last_hidden_state
        else:
            sentence_embedding = self.bert(**inputs).last_hidden_state

        return sentence_embedding


class SBertEmbedder(BaseEmbedder):
    def __init__(self, freeze=True) -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sbert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.freeze = freeze
        self.dim = self.sbert.config.hidden_size
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, to_embed, context=None, device=None, extra_features=None):
        """
        Embed a sentence with the [CLS] token output.
        """
        # Tokenize sentences
        inputs = self.tokenizer(to_embed, padding=True, truncation=True, return_tensors='pt')

        if device is not None:
            inputs = inputs.to(device)
            self.sbert = self.sbert.to(device)

        if self.freeze:
            with torch.no_grad():
                sentence_embedding = self.sbert(**inputs)
        else:
            sentence_embedding = self.sbert(**inputs)

        # Perform pooling
        sentence_embedding = self.mean_pooling(sentence_embedding, inputs['attention_mask'])

        # Normalize embeddings
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding


class ContextBertEmbedder(BaseEmbedder):
    def __init__(self, bert_checkpoint="bert-base-uncased", freeze=True, debug=False) -> None:
        super().__init__()
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        self.bert = AutoModel.from_pretrained(bert_checkpoint)
        self.freeze = freeze
        self.dim = self.bert.config.hidden_size
        self.cache = {}
        self.debug = debug

    def truncate_inputs(self, input, sentence_start=None, sentence_length=None):
        """
        BERT has a window of fixed length, usually 512 tokens. 
        It is often the case that an essay does not fit.
        In this case we shold decide to truncate some context.
        The important things is to keep the original sentece.
        The goal is to remove as less valuable context as possible.
        """
        # TODO: not a solution.
        new_range = range(512)
        for k ,v in input.items():
            input[k] = v[:, :(max(512, v.shape[0]))]
        return input

    
    def forward(self, to_embed, context=None, device=None, extra_features=None):
        n = len(to_embed)
        embeddings = torch.empty((n, self.dim))

        for i, (sentence_i, context_i) in tqdm(enumerate(zip(to_embed, context))):
            if sentence_i in self.cache:
                embeddings[i, :] = self.cache[sentence_i]
                continue

            # drop [CLS] and [SEP]
            tokenized_sentence = self.tokenizer(sentence_i, return_tensors="pt")["input_ids"][0][1:-1]
            context_input = self.tokenizer(context_i, return_tensors="pt")
            context_input = self.truncate_inputs(context_input)
            tokenized_context = context_input["input_ids"][0]
            sentence_start = get_subarray_index(tokenized_context, tokenized_sentence)

            if self.debug:
                init_sentence = self.tokenizer.decode(tokenized_sentence)
                found_sentence = self.tokenizer.decode(
                    tokenized_context[sentence_start:sentence_start+len(tokenized_sentence)]
                )
                if found_sentence != init_sentence:
                    print(f"mismatch: \n init: {init_sentence} \n found: {found_sentence}")

            if self.freeze:
                with torch.no_grad():
                    context_output = self.bert(**context_input)
            else:
                context_output = self.bert(**context_input)

            self.cache[sentence_i] = embeddings[i, :] = torch.mean(
                context_output.last_hidden_state[0, sentence_start:(sentence_start + len(tokenized_sentence)), :],
                dim=0
            )
        
        if device is not None:
            embeddings = embeddings.to(device)

        return embeddings