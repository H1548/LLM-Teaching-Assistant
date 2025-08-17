import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import random
import pickle
from tokenizers import Tokenizer
import math
from torch.cuda.amp import autocast, GradScaler
from utils.py import get_batch, estimate_loss 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUDA_LAUNCH_BLOCKING=1
print(device)

tokenize = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenize.get_vocab_size()
n_embd = 448
n_head = 7
dropout = 0
block_size = 256
batch_size = 32
random.seed(42)


class EncoderHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out=wei @ v 
        return out

class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([EncoderHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): 
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        v = self.value(x)
        out=wei @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out 

class CrossHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y):
        k = self.key(x)
        q = self.query(y)
        

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim = -1)
        v = self.value(x)
        out=wei @ v
        return out 

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        out = torch.cat([h(x,y) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head 
        self.sa = EncoderMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    
    def __init__(self, n_layer):
      
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)
        self.final_norm = nn.LayerNorm(n_embd)
            
        self.layers = nn.Sequential(*[EncoderLayer(n_embd,n_head)
        for _ in range(n_layer)])
                    
    def forward(self, x):
        B,T= x.shape
        tok_emb = self.tok_embeddings(x)
        pos_emb = self.pos_embeddings(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.final_norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size = n_embd//n_head 
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ca = MultiHeadCrossAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
    def forward(self, x, y):
        y = y + self.sa(self.ln1(y))
        y= self.ln2(y)
        y = y + self.ca(x,y)
        y = y + self.ffwd(self.ln3(y))
        return y

class Decoder(nn.Module):
    
    def __init__(self, n_layer):
      
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)
        self.final_norm = nn.LayerNorm(n_embd)
            
        self.layers = nn.ModuleList([DecoderLayer(n_embd,n_head)
        for _ in range(n_layer)])

    def forward(self, x, y):
        B,T= y.shape
        tok_emb = self.tok_embeddings(y)
        pos_emb = self.pos_embeddings(torch.arange(T, device = device))
        y = tok_emb + pos_emb
        for layer in self.layers: 
            y = layer(x,y)
        y = self.final_norm(y)
        return y
        
class Transformer(nn.Module):
    def __init__(self, n_layer):
        super().__init__()
        self.encoder = Encoder(n_layer)
        self.decoder = Decoder(n_layer)
        self.fc = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean = 0.0, std=0.02)
    
    def forward(self, x, y= None):
        enc = self.encoder(x)
        dec = self.decoder(enc,y)
        logits= self.fc(dec)
        
        if y is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            y = y.view(-1)
            loss = F.cross_entropy(logits, y)
            return logits, loss
        else:
            return logits
