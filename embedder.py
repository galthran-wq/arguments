import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer


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


class SBertEmbedder(BaseEmbedder):
    def __init__(self, freeze=True) -> None:
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sbert = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.freeze = freeze
        self.dim = self.sbert.config.hidden_size
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __call__(self, to_embed, context=None, device=None):
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
    def __init__(self, bert_checkpoint="bert-base-uncased", freeze=True) -> None:
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        self.bert = AutoModel.from_pretrained(bert_checkpoint)
        self.freeze = freeze
        self.dim = self.bert.config.hidden_size
        self.cache = {}

    def get_subarray_index(self, A, B):
        n = len(A)
        m = len(B)
        # Two pointers to traverse the arrays
        i = 0
        j = 0
        start = None
    
        # Traverse both arrays simultaneously
        while (i < n and j < m):
    
            # If element matches
            # increment both pointers
            if (A[i] == B[j]):
                if start is None:
                    start = i
    
                i += 1;
                j += 1;
    
                # If array B is completely
                # traversed
                if (j == m):
                    return start;
            
            # If not,
            # increment i and reset j
            else:
                i = i - j + 1;
                j = 0;
            
        return start
    
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

    
    def __call__(self, to_embed, context=None, device=None):
        n = len(to_embed)
        embeddings = torch.empty((n, self.dim))

        for i, (sentence_i, context_i) in enumerate(zip(to_embed, context)):
            if sentence_i in self.cache:
                embeddings[i, :] = self.cache[sentence_i]
                continue

            # drop [CLS] and [SEP]
            tokenized_sentence = self.tokenizer(sentence_i, return_tensors="pt")["input_ids"][0][1:-1]
            context_input = self.tokenizer(context_i, return_tensors="pt")
            context_input = self.truncate_inputs(context_input)
            tokenized_context = context_input["input_ids"][0]
            sentence_start = self.get_subarray_index(tokenized_context, tokenized_sentence)

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