import torch
import torch.nn as nn
from Attention import *
from FeedForward import *
from Norm import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, d_ff, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.h = h
        self.dk = d_k
        self.dv = d_v
        self.dff = d_ff
        self.multi_head_attention = MultiHeadAttention(h, d_model, d_k, d_v)
        self.add_norm = AddNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, X):
        Y = MultiHeadAttention(self.h, self.d_model, self.d_k, self.d_v).forward(X, X, X)
        Y = self.add_norm(X, Y)
        Z = self.feed_forward.forward(Y)
        return self.add_norm(Y, Z)


class Encoder(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_h, d_ff, N, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_h = d_h
        self.d_ff = d_ff
        self.N = N
        self.layers = []
        for i in range(N):
            self.layers.append(EncoderLayer(d_model, d_k, d_v, d_h, d_ff))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, X):
        for i in range(self.N):
            X = self.layers[i].forward(X)
        return X
