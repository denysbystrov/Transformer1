import torch
import torch.nn as nn

from Decoder import *
from Embedding import *

X = torch.rand((1, 60, 32))
d_model = 32
d_k = 8
d_v = 8
d_h = 4
d_ff = 128
N = 6
max_seq_len = 60
embedding = PositionEmbedding(d_model, max_seq_len)
decoder = Decoder(d_model, d_k, d_v, d_h, d_ff, N)

X = embedding(X)
print(X[0])
X = decoder(X)
print(X[0])

