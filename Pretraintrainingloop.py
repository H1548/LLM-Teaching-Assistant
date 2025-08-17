from utils import get_batch, estimate_loss 
from pretrainmodel import Transformer
import math
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_iters = 150000
n_layer = 7
learning_rate = 3e-4
warmup_steps = 12000
current_step = 0
early_stopping_counter = 0
patience = 10
checkpoint = 5000
number = 25000
eval_iters = 500

model = Transformer(n_layer)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(warmup_steps, 1))
    else:
        return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (max_iters - warmup_steps)))
scheduler = LambdaLR(optimizer,lr_lambda)

scaler = GradScaler()
best_val_loss = float('inf')

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss(model)
        print(f"step:{iter}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break
            
    xb, yb = get_batch('train')
    xb,yb  = xb.to(device), yb.to(device)
    
    with autocast():
        logits, loss = model.forward(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    current_step += 1
    if iter > 0 and iter % checkpoint == 0:
        checkpoint_path = f"EncDecCheckpoint/checkpoint_iter_{iter}.pt"
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