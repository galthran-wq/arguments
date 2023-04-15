from typing import List, Dict, Any
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from nltk import word_tokenize

import arguE.data_builder as data_builder
from arguE.data_loader import load_from_directory

from algs import get_subarray_index


class BaseDataset(Dataset):
    @property
    def labels_map(self):
        raise NotImplemented
    
    @property
    def index2label(self):
        return {
            v: k 
            for k, v in self.labels_map.items()
        }

    @property
    def num_labels(self):
        return len(self.labels_map)


class ComponentClassification(BaseDataset):
    """
    Sentence by sentence partition.
    """
    @property
    def labels_map(self):
        return {
            "Premise": 0,
            "Claim": 1,
            "MajorClaim": 2,
        }

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


class ComponentIdentificationAndClassification(BaseDataset):
    """
    Idea: let's try to predict not only where arumentative component is, but also
    what argumentative component this is.

    In Stab, Gurevych (2017) they used two separate models for this.
    They argue that classifying components and estimating argument relations should be done jointly, 
    because there is a lot of shared information and each of the task wouldn't converge to a globally nice solution independently.

        > For instance, if an argument component is classified as claim, it is less likely to exhibit
    outgoing relations and more likely to have incoming relations. On the other hand, an
    argument component with an outgoing relation and few incoming relations is more
    likely to be a premise.

    """
    @property
    def labels_map(self):
        return {
            "O": 0,
            "I-Premise": 1,
            "I-Claim": 2,
            "I-MajorClaim": 3,
            "B-Premise": 4,
            "B-Claim": 5,
            "B-MajorClaim": 6,
        }

    @property
    def num_labels(self):
        return len(self.labels_map)
    
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
        self.ann["component_type"] = self.ann["detail"].map(lambda text: text.split(" ")[0])
        self.arg_components = self.ann[self.ann["type"].str.startswith("T")]
        self.labels = self._get_labels()

        self.index2essay_id = {
            i: essay_id
            for i, essay_id in enumerate(self.essays.keys())
        }
    
    def _get_labels(self):
        labels = {}
        for essay_id, tokenized_essay in self.essays.items():
            label = np.array([ self.labels_map["O"] ] * len(tokenized_essay))

            tokenized_arg_components = self.arg_components.loc[
                self.arg_components["essay_id"] == essay_id,
                ["text", "component_type"]
            ].to_numpy()
            for tokenized_arg_component, component_type in tokenized_arg_components:
                start = get_subarray_index(tokenized_essay, tokenized_arg_component)
                label[start] = self.labels_map[f"B-{component_type}"]
                label[start+1:start+len(tokenized_arg_component)] = self.labels_map[f"I-{component_type}"]
            labels[essay_id] = list(label)
        return labels
        
    def __len__(self):
        return len(self.essays)

    def __getitem__(self, index):
        return (
            self.essays[self.index2essay_id[index]], 
            self.labels[self.index2essay_id[index]]
        )


class ComponentIdentification(ComponentIdentificationAndClassification):
    @property
    def labels_map(self):
        CI_CC_labels_map = super().labels_map
        CI_labels_map = {
            "O": 0,
            "I": 1,
            "B": 2
        }
        # e.g. "B-Premise-> "B"
        remap = {
            k: k[0]
            for k in CI_CC_labels_map.keys()
        }
        backward_labels_map = {
            k: CI_labels_map[v]
            for k, v in remap.items()
        }
        return CI_labels_map | backward_labels_map
    
    @property
    def num_labels(self):
        return 3 


class RelationIdentificationAndClassification(Dataset):
    
    @property
    def labels_map(self):
        return {
            "unrelated": 0,
            "supports": 1,
            "attacks": 2,
        }

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./balanced_relation.csv")
        self.to_embed1 = "arg1"
        self.to_embed2 = "arg2"
        self.pos1 = "pos1"
        self.pos2 = "pos2"
        self.context = "fullText1"
        self.extra_features1 = ["positArg1", "sen1", "tokensArg1",]
        self.extra_features2 = ["positArg2", "sen2", "tokensArg2",]
        self.shared_features = [
            'sharedNouns', 'numberOfSharedNouns', 'sharedVerbs',
            'numberOfSharedVerbs', 'sharedStemWords', 'numberOfSharedStemWords',
            'originalSharedNouns', 'originalNumberOfSharedNouns',
            'originalSharedVerbs', 'originalNumberOfSharedVerbs',
            'originalSharedStemWords', 'originalNumberOfSharedStemWords',
            'sharedStemWordsFull1', 'numberOfSharedStemWordsFull1',
            'sharedStemWordsFull2', 'numberOfSharedStemWordsFull2', 'sameSentence',
            'positionDiff'
        ]
        # filter out 
        self.data = self.data[
            self.data["label"].map(
                lambda label: label in list(self.labels_map.values())
            )
        ]
        self.data.reset_index(inplace=True)
        import json
        self.data[self.pos1] = self.data[self.pos1].map(lambda x: json.loads(x))
        self.data[self.pos2] = self.data[self.pos2].map(lambda x: json.loads(x))

        self.extra_features_dim = len(self.data.loc[0, self.pos1]) + len(self.extra_features1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence1 = self.data.loc[index, self.to_embed1]
        sentence2 = self.data.loc[index, self.to_embed2]
        context = self.data.loc[index, self.context]

        sentence_features1 = self.data.loc[index, self.pos1].copy()
        sentence_features1 += self.data.loc[index, self.extra_features1].tolist()
        sentence_features2 = self.data.loc[index, self.pos2].copy()
        sentence_features2 += self.data.loc[index, self.extra_features2].tolist()

        shared_features = self.data.loc[index, self.shared_features].tolist()
        
        label = int(self.data.loc[index, "label"].item())

        return {
            "sentence1": sentence1, 
            "sentence2": sentence2, 
            "extra_features1": sentence_features1,
            "extra_features2": sentence_features2,
            "shared_features": shared_features,
            "label": label,
            "context": context
        }

class RelationIdentification(RelationIdentificationAndClassification):
    @property
    def labels_map(self):
        return {
            "unrelated": 0,
            "related": 1,
        }
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        super_supports_index = super().labels_map["supports"]
        super_attacks_index = super().labels_map["attacks"]
        self.data["label"] = self.data["label"].replace({
            super_attacks_index: self.labels_map["related"],
            super_supports_index: self.labels_map["related"]
        })


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