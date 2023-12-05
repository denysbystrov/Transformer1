import torch
import torch.nn as nn


class AddNorm(nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.layer_norm(X + Y)
