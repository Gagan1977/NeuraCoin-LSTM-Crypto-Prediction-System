import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architectures.lstm import LSTMModel


def load_test_data(coin_name, time_frame, data_dir='data/processed'):
    '''Load preprocessed test data'''

    file_path = f'{data_dir}/{time_frame}/{coin_name}_data.pt'

    print(f'\n{'='*60}')
    print(f'Loading test data...')
    print(f'File: {file_path}')
    print(f'{'='*60}\n')

    data = torch.load(file_path)

    X_test = data['X_test']
    y_test = data['y_test']

    print(f'Test data:')
    print(f'    X_test: {X_test.shape}')
    print(f'    y_test: {y_test.shape}')

    return X_test, y_test

def load_trained_model(checkpoint_path, input_size, hidden_size, num_layers, output_size, dropout, device):
    '''Load the trained model from checkpoint'''

    print(f'\n{'='*60}')
    print("Loading trained model...")
    print(f'Checkpoint: {checkpoint_path}')
    print(f'{'='*60}\n')

    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )

    model = model.to(device)

    model.load_checkpoint(checkpoint_path)

    model.eval()

    print(f'Model loaded successfully')
    print(f'    Parameters: {model.count_parameters():,}')
    print(f'    Device: {model.get_device()}')

    return model

def make_predictions(model, X_test, device, batch_size=32):
    '''Make predicitions on test set'''

    print(f'\n{'='*60}')
    print('Making predictions on test set...')
    print(f'{'='*60}\n')

    model.eval()

    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch_X = batch[0].to(device)

            pred = model(batch_X)

            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    print(f'Predictions complete')
    print(f'    Shape: {predictions.shape}')

    return predictions

def calculate_metrics(y_true, y_pred, feature_names=['Open', 'High', 'Low', 'Close', 'Volume']):
    '''Calculate evaluation metrics'''

    print(f"\n{'='*60}")
    print("Calculating Metrics...")
    print(f"{'='*60}\n")

    metrics = {}

    for i, feature in enumerate(feature_names):
        true = y_true[:, i]
        pred = y_pred[:, i]

        mae = np.mean(np.abs(true - pred))

        rmse = np.sqrt(np.mean((true - pred) ** 2))

        mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100

        metrics[feature] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        print(f"{feature}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print()

    return metrics

def denormalize_data(data, scaler):
    '''Denormalize data using saved scaler'''

    return scaler.inverse_transform(data)

def plot_predictions(y_true, y_pred, coin_name, timeframe, feature_names=['Open', 'High', 'Low', 'Close', 'Volume'], save_dir='results'):
    '''Plot predictions vs actual values'''

    print(f"\n{'='*60}")
    print("Creating visualizations...")
    print(f"{'='*60}\n")

    os.makedirs(f'{save_dir}/{timeframe}', exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{coin_name.upper()} - {timeframe.capitalize()} Predictions vs Actual', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        ax.plot(y_true[:, i], label='Actual', alpha=0.7, linewidth=2)
        ax.plot(y_pred[:, i], label='Predicted', alpha=0.7, linewidth=2)

        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[5].axis('off')

    plt.tight_layout()

    plot_path = f'{save_dir}/{timeframe}/{coin_name}_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved: {plot_path}')

    plt.close()

def main():
    '''Main evaluation execution'''

    COIN_NAME = 'bitcoin'
    TIMEFRAME = 'daily'

    INPUT_SIZE = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 5
    DROPOUT = 0.2

    CHECKPOINT_PATH = f'model/checkpoints/{TIMEFRAME}/{COIN_NAME}_best.pth'
    SCALER_PATH = f'model/scalers/{TIMEFRAME}/{COIN_NAME}_scaler.pkl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    X_test, y_test = load_test_data(COIN_NAME, TIMEFRAME)

    model = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
        device=device
    )

    predictions_normalized = make_predictions(model, X_test, device)

    print(f'\nLoading scaler from {SCALER_PATH}...')
    scaler = joblib.load(SCALER_PATH)

    # --- ADD THIS DEBUG BLOCK ---
    print("\n--- DEBUGGING SCALER ---")
    print(f"Scaler Data Min: {scaler.data_min_}")
    print(f"Scaler Data Max: {scaler.data_max_}")
    # If Data Max is [1.0, 1.0, 1.0, 1.0, 400000], your scaler is broken.
    # It should be [69000, 69000, 69000, 69000, 100000000] (Real Bitcoin Prices)
    # ----------------------------

    y_test_original = denormalize_data(y_test.numpy(), scaler)
    predictions_original = denormalize_data(predictions_normalized, scaler)

    print('Data denormalized to original scale (USD)')

    metrics = calculate_metrics(y_test_original, predictions_original)

    plot_predictions(y_test_original, predictions_original, COIN_NAME, TIMEFRAME)

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Coin: {COIN_NAME}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Test samples: {len(y_test)}")
    print(f"\nAverage Close Price MAPE: {metrics['Close']['MAPE']:.2f}%")
    print(f"{'='*60}\n")

    print('Evaluation complete!')

if __name__ == '__main__':
    main()