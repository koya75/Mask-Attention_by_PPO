import sys
import numpy as np
import math 
import copy
from time import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#input_size = 1000
#n_layers = 6
#heads = 10


class Transformer(nn.Module):
    def __init__(self, input_size, N, heads):
        super().__init__()
        self.encoder = Encoder(input_size, N, heads)
        self.decoder = Decoder(input_size, N, heads)
        self.out = nn.Linear(input_size, input_size)
    def forward(self, src, trg):
        e_outputs = self.encoder(src)
        d_output = self.decoder(trg, e_outputs)
        output = self.out(d_output)
        return output


class Norm(nn.Module):
    def __init__(self, input_size, eps = 1e-6):
        super().__init__()
    
        self.size = input_size
        
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    """if mask is not None:
        if dec_mask:
            mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))
        else:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)"""
    
    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size):
        super().__init__()
        
        self.input_size = input_size
        self.d_k = input_size // heads
        self.h = heads
        
        self.q_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        
        self.out = nn.Linear(input_size, input_size)
    
    def forward(self, q, k, v):
        
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = attention(q, k, v, self.d_k)

        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.input_size)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, input_size, d_ff=2048):
        super().__init__() 
    
        self.linear_1 = nn.Linear(input_size, d_ff)
        self.linear_2 = nn.Linear(d_ff, input_size)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, input_size, heads):
        super().__init__()
        self.norm_1 = Norm(input_size)
        self.norm_2 = Norm(input_size)
        self.attn = MultiHeadAttention(heads, input_size)
        self.ff = FeedForward(input_size)

    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.attn(x2,x2,x2)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, input_size, heads):
        super().__init__()
        self.norm_1 = Norm(input_size)
        self.norm_2 = Norm(input_size)
        self.norm_3 = Norm(input_size)
        
        self.attn_1 = MultiHeadAttention(heads, input_size)
        self.attn_2 = MultiHeadAttention(heads, input_size)
        self.ff = FeedForward(input_size)

    def forward(self, x, e_outputs):
        x2 = self.norm_1(x)
        x = x + self.attn_1(x2, x2, x2)
        x2 = self.norm_2(x)
        x = x + self.attn_2(x2, e_outputs, e_outputs)
        x2 = self.norm_3(x)
        x = x + self.ff(x2)
        return x

"""class PositionalEncoder(nn.Module):
    def __init__(self, input_size, max_seq_len = 1):
        super().__init__()
        self.input_size = input_size
        pe = torch.zeros(max_seq_len, input_size)
        for pos in range(max_seq_len):
            for i in range(0, input_size, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/input_size)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/input_size)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        x = x * math.sqrt(self.input_size)
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return x"""

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, input_size, N, heads):
        super().__init__()
        self.N = N
        #self.pe = PositionalEncoder(input_size)
        self.layers = get_clones(EncoderLayer(input_size, heads), N)
        self.norm = Norm(input_size)
    def forward(self, src):
        x = src
        for i in range(self.N):
            x = self.layers[i](x)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, input_size, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(input_size, heads), N)
        self.norm = Norm(input_size)
    def forward(self, trg, e_outputs):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.norm(x)
