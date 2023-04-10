from typing import List, Dict, Any
import torch
import numpy as np
import pandas as pd
from arguE.data_loader import load_from_directory
from torch.utils.data import Dataset
from nltk import word_tokenize

import arguE.data_builder as data_builder


class ComponentClassification(Dataset):
    """
    Sentence by sentence partition.
    """
    def __init__(self) -> None:
        super().__init__()
        self.data = load_from_directory(
            "./arguEParser/outputCorpora/essays/",
            ADU=True
        )
        self.data = data_builder.add_features(self.data, has_2=False) 
        self.feature_columns = [
            'positArg1',
            'tokensArg1', 'sharedStemWordsFull1',
            'numberOfSharedStemWordsFull1'
        ]
        self.n_features = len(self.data.loc[0, "pos1"]) + len(self.feature_columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns:
        dict
            sentence, -- str 
            context, -- str
            sentence_features -- vector of arguE features about the sentence
                Includes position, POS, shared with full text features
            label -- 0, 1, 2 for premise, claim, and major claim
        """
        sentence = self.data.loc[index, "originalArg1"]
        context = self.data.loc[index, "fullText1"]
        label = self.data.loc[index, "label"].item()
        sentence_features = self.data.loc[index, "pos1"].copy()
        sentence_features += self.data.loc[index, self.feature_columns].tolist()

        return {
            "sentence": sentence, 
            "context": context,
            "sentence_features": sentence_features,
            "label": label
        }


class ComponentIdentification(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.ann = pd.DataFrame(columns=["essay_id", "type", "detail", "text"])
        self.essays = {}

        for i in range(1, 403):
            essay_ann_i = pd.read_csv(
                f"./arguEParser/inputCorpora/essays/essay{'%03d' % i }.ann", 
                delimiter="\t", header=None
            )
            with open(f"./arguEParser/inputCorpora/essays/essay{'%03d' % i }.txt", "r") as f:
                essay_i = f.read()

            essay_ann_i.columns = ["type", "detail", "text"]
            essay_ann_i["essay_id"] = i
            self.ann = pd.concat([self.ann, essay_ann_i])
            self.essays[i] = essay_i

        self.essays = { k: word_tokenize(v) for k, v in self.essays.items() }
        self.ann.reset_index(inplace=True)
        self.ann["text"] = self.ann["text"].map(lambda text: word_tokenize(text) if isinstance(text, str) else text)
        self.arg_components = self.ann[self.ann["type"].str.startswith("T")]
        self.labels = self._get_labels()

        self.index2essay_id = {
            i: essay_id
            for i, essay_id in enumerate(self.essays.keys())
        }
    
    @property
    def num_labels(self):
        return 3
    
    def get_subarray_index(self, A, B):
        n = len(A)
        m = len(B)
        # Two pointers to traverse the arrays
        i = 0
        j = 0
        best_match_length = 0
        match_length = 0
        best_start = None
        start = None
    
        # Traverse both arrays simultaneously
        while (i < n and j < m):
    
            # If element matches
            # increment both pointers
            if (A[i] == B[j]):
                if start is None:
                    start = i
                
                match_length += 1
                i += 1;
                j += 1;
    
                # If array B is completely
                # traversed
                if (j == m):
                    if match_length > best_match_length:
                        best_match_length = match_length
                        best_start = start
                    return best_start;
            
            # If not,
            # increment i and reset j
            else:
                if match_length > best_match_length:
                    best_match_length = match_length
                    best_start = start
                start = None
                match_length = 0
                i = i - j + 1;
                j = 0;
            
        if match_length > best_match_length:
            best_match_length = match_length
            best_start = start
        return best_start;
    
    def _get_labels(self):
        labels = {}
        for essay_id, tokenized_essay in self.essays.items():
            label = np.array([ 0 ] * len(tokenized_essay))

            tokenized_arg_components = self.arg_components.loc[
                self.arg_components["essay_id"] == essay_id,
                "text"
            ].tolist()
            for tokenized_arg_component in tokenized_arg_components:
                start = self.get_subarray_index(tokenized_essay, tokenized_arg_component)
                label[start] = 2
                label[start+1:start+len(tokenized_arg_component)] = 1
            labels[essay_id] = list(label)
        return labels
        
    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        return (
            self.essays[self.index2essay_id[index]], 
            self.labels[self.index2essay_id[index]]
        )


class DictionaryCollator:
    """
    Inspired by transformers/data/data_collator.py
    """
    def __call__(self, features: List[Dict[str, Any]], return_tensors="pt") -> Dict[str, Any]:
        first = features[0]
        batch = {}

        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        
        for k, v in first.items():
            if isinstance(v, str):
                batch[k] = [f[k] for f in features]
            elif k != "label" and v is not None:
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
        return batch