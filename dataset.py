from typing import List, Dict, Any
import torch
import numpy as np
from transformers import BertTokenizer
from arguE.data_loader import load_from_directory
from torch.utils.data import Dataset

import arguE.data_builder as data_builder


class ComponentIdentification(Dataset):
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
        label = self.data.loc[index, "label"]
        sentence_features = self.data.loc[index, "pos1"].copy()
        sentence_features += self.data.loc[index, 
            [
            'positArg1',
            'tokensArg1', 'sharedStemWordsFull1',
            'numberOfSharedStemWordsFull1'
            ]
        ].tolist()

        return {
            "sentence": sentence, 
            "context": context,
            "sentence_features": sentence_features,
            "label": label
        }


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