import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.lstm import LSTMModel


def load_processed_data(coin_name, timeframe, data_dir='data/processed'):
    '''Load preprocessed data from disk'''

    file_path = f'{data_dir}/{timeframe}/{coin_name}_data.pt'

    print(f'\n{'='*60}')
    print(f'Loading preprocessed data...')
    print(f'File: {file_path}')
    print(f'{'='*60}\n')

    data = torch.load(file_path)

    print("Loaded tensors:")
    print(f'    X_train: {data['X_train'].shape}')
    print(f'    y_train: {data['y_train'].shape}')
    print(f'    X_val: {data['X_val'].shape}')
    print(f'    y_val: {data['y_val'].shape}')
    print(f'    X_test: {data['X_test'].shape}')
    print(f'    y_test: {data['y_test'].shape}')

    return data

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    '''Create DataLoaders for training and validation'''

    print(f'\nCreating DataLoaders with batch_size={batch_size}...')

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    print(f'    Train batches: {len(train_loader)}')
    print(f'    Val batches: {len(val_loader)}')

    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    '''Train model for one epoch'''

    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_X)

        loss = criterion(predictions, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss

def validate(model, val_loader, criterion, device):
    '''Evaluate model on validation set'''

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    return avg_loss