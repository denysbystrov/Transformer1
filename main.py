# implementation of a transformer model for sentence completion using pytorch

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, d_model, d_k, d_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.output = None

    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        # compute the scaled dot-product attention

        SM = nn.Softmax(dim=2)
        QK = torch.bmm(Q, K.transpose(1, 2))
        QK = QK / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        QK = SM(QK)
        self.output = torch.bmm(QK, V)
        return self.output


# implement the multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention_heads = []
        for i in range(h):
            self.attention_heads.append(AttentionHead(d_model, d_k, d_v))
        self.W_O = nn.Linear(h * d_v, d_model)

    def forward(self, Q, K, V):
        # concatenate the outputs of the attention heads into one matrix
        # and apply a linear transformation to get the final output
        outputs = []
        for i in range(self.h):
            self.attention_heads[i].forward(Q, K, V)
            outputs.append(self.attention_heads[i].output)
            outputs = torch.cat(outputs, dim=2)
        outputs = self.W_O(outputs)
        return outputs


class AddNorm(nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X, Y):
        return self.layer_norm(X + Y)


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




