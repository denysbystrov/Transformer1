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
        mask = torch.tril(torch.ones(Q.shape[1], K.shape[1]))
        SM = nn.Softmax(dim=2)
        QK = Q @ K.transpose(1, 2)
        QK = QK / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        QK = QK.masked_fill(mask == 0, float('-inf'))
        QK = SM(QK)
        return QK @ V


# implement the multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.attention_heads = nn.ModuleList([AttentionHead(d_model, d_k, d_v) for _ in range(h)])
        self.W_O = nn.Linear(h * d_v, d_model)

    def forward(self, Q, K, V):
        # concatenate the outputs of the attention heads into one matrix
        # and apply a linear transformation to get the final output
        outputs = torch.cat([head(Q, K, V) for head in self.attention_heads], dim=2)
        outputs = self.W_O(outputs)
        return outputs
