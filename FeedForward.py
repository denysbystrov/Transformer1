import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        return self.linear_2(torch.relu(self.linear_1(X) + self.b1)) + self.b2
