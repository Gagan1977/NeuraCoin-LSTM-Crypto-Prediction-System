import torch
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
import os
import sys
import argparse


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architectures.lstm import LSTMModel

def fetch_binance_data(symbol='BTCUSDT', interval='1d', limit=60):
    '''Fetch recent data from Binance API'''

    print(f"\n{'='*60}")
    print("Fetching data from Binance...")
    print(f"{'='*60}\n")

    url = 'https://api.binance.com/api/v3/klines'

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df = df[['open', 'high', 'low', 'close', 'volume']]

        df = df.astype(float)

        print(f'Fetched {len(df)} candles')
        print(f'    Latest Close: ${df['close'].iloc[-1]:,.2f}')

        return df.values
    
    except Exception as e:
        print(f'Error fetching data: {e}')
        return None
    
def predict_target_day(model, initial_data, scaler, days_ahead, device):
    '''Predict target day only'''

    print(f"\n{'='*60}")
    print(f"Predicting {days_ahead} day(s) ahead...")
    print(f"{'='*60}\n")

    model.eval()
    current_sequence = initial_data.copy()

    with torch.no_grad():
        for day in range(days_ahead):
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            prediction = model(input_tensor).cpu().numpy()[0]
            current_sequence = np.vstack([current_sequence[1:], prediction])

        final_prediction = scaler.inverse_transform(prediction.reshape(1, -1))[0]

        print(f'Prediction complete!')

        return final_prediction
    
def display_prediction(prediction, coin_name, days_ahead, timeframe):
    '''Display prediction in a clean formt'''

    if timeframe == 'daily':
        target_date = datetime.now() + timedelta(days=days_ahead)
        ahead_text = f'{days_ahead} day(s)'
    else:
        target_date = datetime.now() + timedelta(hours=days_ahead)
        ahead_text = f'{days_ahead} hour(s)'

    print(f"\n{'='*60}")
    print(f"{coin_name.upper()} PRICE PREDICTION")
    print(f"{'='*60}")
    print(f"Target Date: {target_date.strftime('%B %d, %Y %H:%M')}")
    print(f"({ahead_text} from now)")
    print(f"{'='*60}\n")

    print(f"PREDICTED VALUES (USD):\n")
    print(f"    Open:   ${prediction[0]:,.2f}")
    print(f"    High:   ${prediction[1]:,.2f}")
    print(f"    Low:    ${prediction[2]:,.2f}")
    print(f"    Close:  ${prediction[3]:,.2f}")
    print(f"    Volume: {prediction[4]:,.0f}")

    print(f"\n{'='*60}")

    if timeframe == 'daily':
        if days_ahead <= 3:
            print("Short-term prediction (good accuracy expected)")
        elif days_ahead <= 7:
            print("Medium-term prediction (moderate accuracy)")
        else:
            print("Long-term prediction (lower accuracy - use with caution)")
    else:
        if days_ahead <= 24:
            print("Short-term prediction (good accuracy expected)")
        else:
            print("Long-term prediction (lower accuracy)")

    print(f'{'='*60}\n')

def main():
    '''Main prediction execution'''

    parser = argparse.ArgumentParser(description='Predict cryptocurrency prices')
    parser.add_argument('--coin', type=str, default='bitcoin', help='Coin name (e.g., bitcoin)')
    parser.add_argument('--timeframe', type=str, default='daily', choices=['daily', 'hourly'], help='Timeframe')
    parser.add_argument('--days', type=int, default=1, help='Days ahead to predict')

    args = parser.parse_args()

    COIN_NAME = args.coin
    TIMEFRAME = args.timeframe
    DAYS_AHEAD = args.days

    INPUT_SIZE = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    OUTPUT_SIZE = 5
    DROPOUT = 0.2

    CHECKPOINT_PATH = f'model/checkpoints/{TIMEFRAME}/{COIN_NAME}_best.pth'
    SCALER_PATH = f'model/scalers/{TIMEFRAME}/{COIN_NAME}_scaler.pkl'

    SYMBOL_MAP = {
        'bitcoin': 'BTCUSDT',
        'ethereum': 'ETHUSDT',
        'cardano': 'ADAUSDT',
        'solana': 'SOLUSDT',
        'ripple': 'XRPUSDT',
        'polkadot': 'DOTUSDT',
        'dogecoin': 'DOGEUSDT',
        'litecoin': 'LTCUSDT'
    }

    INTERVAL = '1d' if TIMEFRAME == 'daily' else '1h'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"CRYPTOCURRENCY PRICE PREDICTION")
    print(f"{'='*60}")
    print(f"Coin: {COIN_NAME.upper()}")
    print(f"Timeframe: {TIMEFRAME.capitalize()}")
    print(f"Predicting: {DAYS_AHEAD} {TIMEFRAME} ahead")
    print(f"Device: {device}")
    print(f"{'='*60}")

    symbol = SYMBOL_MAP.get(COIN_NAME.lower())
    if not symbol:
        print(f'Unknown coin: {COIN_NAME}')
        return
    
    fresh_data = fetch_binance_data(symbol=symbol, interval=INTERVAL, limit=60)
    
    if fresh_data is None:
        print("Failed to fetch data from Binance")
        return
    
    print(f'\nLoading scaler from {SCALER_PATH}')
    
    if not os.path.exists(SCALER_PATH):
        print(f'Scaler not found! Please train the model first.')
        return
    
    scaler = joblib.load(SCALER_PATH)
    print('Scaler loaded')

    fresh_data_normalized = scaler.transform(fresh_data)

    print(F'\nLoading model from {CHECKPOINT_PATH}...')

    if not os.path.exists(CHECKPOINT_PATH):
        print(f'Model not found! Please train the model first')
        return
    
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    )

    model = model.to(device)
    model.load_checkpoint(CHECKPOINT_PATH)
    model.eval()

    print(f'Model loaded')

    prediction = predict_target_day(
        model=model,
        initial_data=fresh_data_normalized,
        scaler=scaler,
        days_ahead=DAYS_AHEAD,
        device=device
    )

    display_prediction(prediction, COIN_NAME, DAYS_AHEAD, TIMEFRAME)


if __name__ == '__main__':
    main()