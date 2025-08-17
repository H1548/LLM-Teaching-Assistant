import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import mmap
import random
import pickle
import argparse
from tokenizers import Tokenizer
import math
from torch.cuda.amp import autocast, GradScaler
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CUDA_LAUNCH_BLOCKING=1
print(device)

tokenize = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenize.get_vocab_size()
n_embd = 448
learning_rate =1e-6
max_iters = 100000
eval_iters = 500
n_layer = 7
n_head = 7
dropout = 0
block_size = 256
batch_size = 28
warmup_steps = 12000
current_step = 0
early_stopping_counter = 0
patience = 10
checkpoint = 5000
number = 25000
max_length = 256
random.seed(42)
Sep_token = tokenize.token_to_id('[SEP]')
CLS_token = tokenize.token_to_id('[CLS]')
Pad_token = tokenize.token_to_id('[PAD]')


def pad_sequence(sequence, max_length, pad_token_id):
    # Truncate the sequence if it's longer than max_length
    if len(sequence) > max_length:
        return sequence[:max_length]
    # Pad the sequence if it's shorter than max_length
    return sequence + [pad_token_id] * (max_length - len(sequence))

def create_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()


class EncoderHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask = None): 
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            wei= wei.masked_fill(mask == 0, float('-inf'))
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
        
    def forward(self, x, mask = None):
        out = torch.cat([h(x, mask) for h in self.heads], dim = -1)
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
        
    def forward(self, x, mask = None): 
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            wei= wei.masked_fill(mask == 0, float('-inf'))
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
        
    def forward(self, x, mask = None):
        out = torch.cat([h(x, mask) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out 

class CrossHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y, mask = None):
        k = self.key(x)
        q = self.query(y)
        

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            wei= wei.masked_fill(mask == 0, float('-inf'))
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
        
    def forward(self, x, y, mask = None):
        out = torch.cat([h(x,y, mask) for h in self.heads], dim = -1)
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
    def forward(self, x, mask = None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class Encoder(nn.Module):
    
    def __init__(self, n_layer):
      
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.pos_embeddings = nn.Embedding(block_size, n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)
        self.final_norm = nn.LayerNorm(n_embd)
            
        self.layers = nn.ModuleList([EncoderLayer(n_embd,n_head)
        for _ in range(n_layer)])
                    
    def forward(self, x, mask = None):
        B,T= x.shape
        tok_emb = self.tok_embeddings(x)
        pos_emb = self.pos_embeddings(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        for layer in self.layers: 
            x = layer(x,mask)
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
    def forward(self, x, y, mask = None,target_mask = None):
        
        y = y + self.sa(self.ln1(y),target_mask)
        
        y= self.ln2(y)
        y = y + self.ca(x,y,mask)
        y = self.ln3(y)
        y = y + self.ffwd(y)
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

    def forward(self, x, y, mask = None, target_mask = None):
        B,T= y.shape
        tok_emb = self.tok_embeddings(y)
        pos_emb = self.pos_embeddings(torch.arange(T, device = device))
        y = tok_emb + pos_emb
        for layer in self.layers: 
            y = layer(x,y, mask, target_mask)
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
    
    def forward(self, x, y, mask = None, target_mask = None):
        enc = self.encoder(x, mask)
        
        dec = self.decoder(enc,y, mask, target_mask)
        logits= self.fc(dec)
        
        B,T,C = logits.shape
        logits = logits.view(B*T,C)
        y = y.view(-1)
        loss = F.cross_entropy(logits, y)
        logits = logits.view(B,T,C)
        return logits, loss
    def generate_response(self,prompt):
     
     tokenized_input = tokenize.encode(prompt)
     input_ids = tokenized_input.ids
     input_ids = pad_sequence(input_ids ,block_size, Pad_token)
     input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
     
     target_tensor = torch.full((1,1),CLS_token,dtype = torch.long).to(device)
     text = []
     self.eval()  # Set the model to evaluation mode
     with torch.no_grad():
        # attention_mask_input = create_attention_mask(input_tensor, Pad_token)
        # target_mask_input = create_attention_mask(target_tensor, Pad_token)
        
        for _ in range(max_length):
            attention_mask = create_attention_mask(input_tensor, Pad_token)
            logits, _ = self.forward(input_tensor, target_tensor, attention_mask)
            #print(logits.shape)
            logits = logits [:, -1,:]
            logits = F.softmax(logits, dim = -1)
            target_tensor_next = torch.multinomial(logits, 1)
            
            text.append(target_tensor_next)
            target_tensor = torch.cat((target_tensor, target_tensor_next), dim=1)
            print(target_tensor)
            if target_tensor_next == Sep_token:
                break
        text = tokenize.decode(target_tensor[0].tolist())
        return text

model = Transformer(n_layer)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(warmup_steps, 1))
    else:
        return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (max_iters - warmup_steps)))
scheduler = LambdaLR(optimizer,lr_lambda)


scaler = GradScaler()
parameters = torch.load('Fine-TuneCheckpoint\checkpoint_iter_10000.pt')
model.load_state_dict(parameters['model_state_dict'])

optimizer.load_state_dict(parameters['optimizer_state_dict'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(device)
            
scaler.load_state_dict(parameters['scaler_state_dict'])
scheduler.load_state_dict(parameters['scheduler_state_dict'])
last_iter = parameters['iter'] if 'iter' in parameters else 0
current_step = last_iter
best_val_loss = parameters['best_val_loss'] if 'best_val_loss' in parameters else float('inf')
early_stopping_counter = parameters['early_stopping_counter'] if 'early_stopping_counter' in parameters else 0     

model = model.to(device)
    
while True:
    input_ids= input("Prompt: \n")
    generated_sequence =  model.generate_response(input_ids)
    print(f"Generated Sequence : {generated_sequence}")