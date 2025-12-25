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

        optimizer.zero_grad()

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, checkpoint_path):
    '''Main training loop with early stopping'''

    print(f'\n{'='*60}')
    print(f'Starting Training')
    print('=' * 60)
    print(f'Device: {device}')
    print(f'Epochs: {num_epochs}')
    print(f'Patience: {patience}')
    print(F'Checkpoint: {checkpoint_path}')
    print(f'{'='*60}\n')

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('=' * 40)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Train Loss: {train_loss: .6f}')
        print(f'Val Loss: {val_loss: .6f}')

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            print(f'Validation loss improved! Saving model...')
            model.save_checkpoint(
                path=checkpoint_path,
                epoch=epoch + 1,
                optimizer=optimizer,
                loss=val_loss
            )
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement} epoch(s)')

        print()

        if epochs_without_improvement >= patience:
            print(f'\n{'='*60}')
            print(f'Early stopping triggered!')
            print(f'No improvement for {patience} consecutive epochs')
            print(f'Best validation loss: {best_val_loss: .6f}')
            print(f'{'='*60}\n')
            break

    print(f'\n{'='*60}')
    print('Training Complete!')
    print(f'Best Validation Loss: {best_val_loss: .6f}')
    print(f'{'='*60}\n')

    return history

def main():
    '''Main training execution'''

    COIN_NAME = 'bitcoin'
    TIMEFRAME = 'daily'

    INPUT_SIZE = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 5
    DROPOUT = 0.2

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15

    CHECKPOINT_DIR = 'model/checkpoints'
    CHECKPOINT_PATH =  f'{CHECKPOINT_DIR}/{TIMEFRAME}/{COIN_NAME}_best.pth'

    os.makedirs(f'{CHECKPOINT_DIR}/{TIMEFRAME}', exist_ok=True)

    data = load_processed_data(COIN_NAME, TIMEFRAME)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
    )

    print(f'\n{'='*60}')
    print('Initializing Model')
    print(f'{'='*60}\n')

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    )

    print(f'Model: LSTMModel')
    print(f'    Input Size: {INPUT_SIZE}')
    print(f'    Hidden size: {HIDDEN_SIZE}')
    print(f'    Num layers: {NUM_LAYERS}')
    print(f'    Output size: {OUTPUT_SIZE}')
    print(f'    Dropout: {DROPOUT}')
    print(f'    Total parameters: {model.count_parameters():,}')

    if torch.cuda.is_available():
        model = model.to('cuda')
        print('GPU available! Using CUDA')
    else:
        model = model.to('cpu')
        print(f'GPU not available. Using CPU')

    device = model.get_device()
    print(f'Model device: {device}')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nLoss Function: MSELoss")
    print(f"Optimizer: Adam")
    print(f"Learning Rate: {LEARNING_RATE}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=CHECKPOINT_PATH
    )

    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Coin: {COIN_NAME}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")
    print(f"{'='*60}\n")

    print('Training complete! You can now evaluate the model using evaluate.py')

if __name__ == '__main__':
    main()