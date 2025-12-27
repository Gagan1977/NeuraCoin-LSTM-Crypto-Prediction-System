import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_csv(file_path):
    '''Load CryptoCurrency data from CSV'''

    print(f'Loading data from {file_path}...')
    df = pd.read_csv(file_path)

    if 'Date' in df.columns:
        print("Sorting data by Date (Oldest -> Newest)...")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    else:
        if df['Close'].iloc[0] > df['Close'].iloc[-1] * 100:
             print("Data appears to be Newest-First. Reversing...")
             df = df.iloc[::-1].reset_index(drop=True)

    print(f'Loaded {len(df)} rows')
    print(f'Columns: {df.columns.tolist()}')

    return df

def extract_features(df):
    '''Extract OHLCV features, drop Date and Instrument'''

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    print(f'Extracted features shape: {data.shape}')

    return data

def normalize_data(data):
    '''Normalize data to 0-1 range using MinMaxScaler'''

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print(f'Data normalized to range [{scaled_data.min(): .2f}, {scaled_data.max(): .2f}]')

    return scaled_data, scaler

def create_sequences(data, sequence_length=60):
    '''Create sequences for time series prediction'''

    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    print(f'Created {len(X)} sequences')
    print(f'X shape: {X.shape}, y shape: {y.shape}')

    return X, y

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    '''Split data into train, validation, and test sets'''

    n = len(X)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    print(f'Train: {len(X_train)} samples')
    print(f'Val: {len(X_val)} samples')
    print(f'Test: {len(X_test)} samples')

    return X_train, y_train, X_val, y_val, X_test, y_test

def to_tensors(X_train, y_train, X_val, y_val, X_test, y_test):
    '''Convert numpy arrays to PyTorch tensors'''

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    print('Converted to PyTorch tensors')

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler, coin_name, timeframe, output_dir='data/processed'):
    '''Save processed data and scaler'''

    os.makedirs(f'{output_dir}/{timeframe}', exist_ok=True)
    os.makedirs(f'model/scalers/{timeframe}', exist_ok=True)

    torch.save({
        'X_train' : X_train,
        'y_train' : y_train,
        'X_val' : X_val,
        'y_val' : y_val,
        'X_test' : X_test,
        'y_test' : y_test
        }, f'{output_dir}/{timeframe}/{coin_name}_data.pt'
    )
    
    joblib.dump(scaler, f'model/scalers/{timeframe}/{coin_name}_scaler.pkl')

    print(f'\nSaved processed data to {output_dir}/{timeframe}/{coin_name}_data.pt')
    print(f'Saved scaler to model/scalers/{timeframe}/{coin_name}_scaler.pkl')

def preprocess_pipeline(csv_path, coin_name, timeframe, sequence_length=60):
    '''Complete preprocessing pipeline'''

    print(f'\n{'='*60}')
    print(f'Preprocessing {coin_name} {timeframe} data')
    print(f'\n{'='*60}')

    df = load_csv(csv_path)

    data = extract_features(df)

    scaled_data, scaler = normalize_data(data)

    X, y = create_sequences(scaled_data, sequence_length)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    X_train, y_train, X_val, y_val, X_test, y_test = to_tensors(X_train, y_train, X_val, y_val, X_test, y_test)

    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler, coin_name, timeframe)

    print(f'\n{'='*60}')
    print(f'Preprocessing complete for {coin_name} {timeframe}')
    print(f'\n{'='*60}')


if __name__ == '__main__':

    preprocess_pipeline(
        csv_path='data/daily_data/bitcoin_daily.csv',
        coin_name='bitcoin',
        timeframe='daily',
        sequence_length=60
    )