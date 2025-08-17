import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import mmap
from tokenizers import Tokenizer
from FineTuneDataLoader.py import load_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenize = Tokenizer.from_file("tokenizer.json")

block_size = 256
batch_size = 32
eval_iters = 500
Mask_token = torch.tensor([tokenize.token_to_id('[MASK]')])
CLS_token = torch.tensor([tokenize.token_to_id("[CLS]")])
SEP_token = torch.tensor([tokenize.token_to_id("[SEP]")])
Pad_token = tokenize.token_to_id('[PAD]')
dataset_path = 'Exam questions - Fine-tuning test.json'

def get_random_chunk(split):
    filename = "train_split.txt" if split =='train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access =mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0,(file_size)- block_size*batch_size)
            
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            encoded_block = tokenize.encode(decoded_block)
            data = torch.tensor(encoded_block.ids, dtype=torch.long)
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([torch.cat((CLS_token,data[i+1:i+block_size+1],SEP_token))for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            X,Y = X.to(device), Y.to(device)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def pad_sequence(sequence, max_length, pad_token_id):
    # Truncate the sequence if it's longer than max_length
    if len(sequence) > max_length:
        return sequence[:max_length]
    # Pad the sequence if it's shorter than max_length
    return sequence + [pad_token_id] * (max_length - len(sequence))

def create_attention_mask(input_ids, pad_token_id):
    return (input_ids != pad_token_id).long()

def get_fine_batch(split):
    train_data, val_data = load_data()
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    
    x, y = [], []

    for i in ix:
        input_seq, output_seq = data[i]
        
        input_seq_padded = pad_sequence(input_seq, block_size, Pad_token)
        output_seq_padded = pad_sequence(output_seq, block_size, Pad_token)

        x.append(torch.tensor(input_seq_padded,dtype=torch.long))
        y.append(torch.tensor(output_seq_padded,dtype=torch.long))

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    attention_masks = [create_attention_mask(seq, Pad_token) for seq in x]
    output_masks = [create_attention_mask(seq, Pad_token) for seq in y]

    attention_masks = torch.stack(attention_masks).to(device)
    output_masks = torch.stack(output_masks).to(device)
    
    return x, y, attention_masks, output_masks

@torch.no_grad()
def estimate_loss_fine(model):
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y, attention_mask, output_mask = get_fine_batch(split)
            X,Y = X.to(device), Y.to(device)
            logits, loss = model(X,Y, attention_mask, output_mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out