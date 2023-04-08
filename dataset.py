import torch
from transformers import BertTokenizer
from data_loader import load_from_directory
from torch.utils.data import Dataset


class ComponentIdentification(Dataset):
    """
    Sentence by sentence partition.
    """
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data = load_from_directory(
            "./arguEParser/outputCorpora/essays/",
            ADU=True
        ).loc[:, ["originalArg1", "label"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data.iloc[index, :]
        return { 
            "sentence": self.tokenizer(entry["originalArg1"], return_tensors="pt")["input_ids"].squeeze(0), 
            "label": torch.tensor(entry["label"])
        }


class ComponentIdentification2(Dataset):
    """
    Sentence by sentence partition.
    """
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data = load_from_directory(
            "./arguEParser/outputCorpora/essays/",
            ADU=True
        ).loc[:, ["originalArg1", "label"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.loc[index, "originalArg1"], self.data.loc[index, "label"]
