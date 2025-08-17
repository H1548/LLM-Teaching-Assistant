import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
import random
import pickle
import math
from modelfinetuning.py import Transformer
from utils import get_fine_batch, estimate_loss_fine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_iters = 10000
eval_iters = 500
learning_rate =1e-6
n_layer = 7
warmup_steps = 12000
current_step = 0
early_stopping_counter = 0
patience = 10
checkpoint = 5000
number = 25000



model = Transformer(n_layer)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(warmup_steps, 1))
    else:
        return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (max_iters - warmup_steps)))
scheduler = LambdaLR(optimizer,lr_lambda)



scaler = GradScaler()

parameters = torch.load('EncDecCheckpoint\checkpoint_iter_10000.pt')
model.load_state_dict(parameters['model_state_dict'])
scaler.load_state_dict(parameters['scaler_state_dict'])
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

for iter in range(max_iters):
    if iter % eval_iters ==0:
        losses = estimate_loss_fine()
        print(f"step:{iter}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break
            
    xb, yb, attention_maskb, output_maskb = get_fine_batch('train')
    xb,yb, attention_maskb, output_maskb  = xb.to(device), yb.to(device), attention_maskb.to(device), output_maskb.to(device)
    
    with autocast():
        logits, loss = model.forward(xb, yb, attention_maskb, output_maskb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    current_step += 1
    if iter > 0 and iter % checkpoint == 0:
        checkpoint_path = f"Fine-TuneCheckpoint/checkpoint_iter_{iter}.pt"
        torch.save({
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'current_loss': loss.item(),
            'best_val_loss': best_val_loss,
            'early_stopping_counter': early_stopping_counter,
        }, checkpoint_path)
        print(f'Periodic checkpoint saved at iteration {iter}')
print(loss.item())
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)