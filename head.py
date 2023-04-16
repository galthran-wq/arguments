from typing import Any
import lightning
import lightning.pytorch as pl
import torch.nn as nn

class BaseHead(nn.Module):
    def __init__(
        self, *args: Any, 
        input_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        dropout_decay: float = 1.,
        n_hidden_layers: int = 0,
        **kwargs: Any
    ) -> None:
        """AI is creating summary for __init__

        Args:
            input_dim (int):
            output_dim (int):
            dropout (float, optional): Defaults to 0.5.
            dropout_decay (float, optional): Defaults to 1..
            n_hidden_layers (int, optional): Defaults to 0.
        """
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.dropout_decay = dropout_decay

        self.hidden_layers = nn.Sequential()

        self.output_layer= nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )
    
    def forward(self, x):
        return self.output_layer(x)