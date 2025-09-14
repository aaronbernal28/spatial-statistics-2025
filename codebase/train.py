import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dataset_token(Dataset):
    def __init__(self, x, y):
        x = False

    def __len__(self):
        return None
    
    def __getitem__(self, idx):
        return None

def custom_collate(batch):
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)  # (batch, max_len, 768)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input': padded_inputs,
            'target': padded_targets,
            'input_length': torch.tensor([item['input_length'] for item in batch]),
            'target_length': torch.tensor([item['target_length'] for item in batch])}

def plot_losses(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    plt.xlabel('Steps')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_steps(model, train_loader, val_loader=None, max_steps=1000, lr=1e-3, verbose_each=50):
    """
    Basado en step - printea cada verbose_each steps
    """
    train_losses = []
    val_losses = []
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Iniciando entrenamiento...")
    print("-" * 50)
    start_time = time.time()
    
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    step = 0
    current_val_loss = 0
    while step < max_steps:
        model.train()
        
        try:
            batch = next(train_iterator)
        except StopIteration:
            # reinicia la iteracion por los batches
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        
        input = batch['input']
        target = batch['target']

        optimizer.zero_grad()
        loss = model.loss(input, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        step += 1
        
        current_train_loss = loss.item()
        train_losses.append(current_train_loss)
        
        # Validacion
        if val_loader is not None:
            model.eval()
            
            with torch.no_grad():
                try:
                    batch_val = next(val_iterator)
                except StopIteration:
                    # reinicia la iteracion por los batches
                    val_iterator = iter(val_loader)
                    batch_val = next(val_iterator)
                input = batch_val['input']
                target = batch_val['target']
                loss_val = model.loss(input, target)
                current_val_loss = loss_val.item()
                val_losses.append(current_val_loss)
                
        # Progreso cada verbose_each
        if step % verbose_each == 0:
            if val_loader is not None:
                print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {current_train_loss:.4f} | Pérdida Validación: {current_val_loss:.4f}', end='\r')
            else:
                print(f'Paso {step}/{max_steps}: Pérdida Entrenamiento: {current_train_loss:.4f}', end='\r')
    
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Entrenamiento completado! Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    
    return train_losses, val_losses