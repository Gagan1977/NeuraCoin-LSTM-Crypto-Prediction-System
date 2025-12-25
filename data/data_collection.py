import os
import requests
import json
import time
import pandas as pd
from forex_python.converter import CurrencyRates
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read and assign the loaded data from .env
API_KEY = os.getenv('COINDESK_API_KEY')
MARKET = os.getenv('COINDESK_MARKET')
LIMIT = int(os.getenv('DEFAULT_LIMIT'))

if not API_KEY:
    raise RuntimeError('COINDESK_API_KEY not found.')


def load_crypto_currencies():

    '''
    Load Crypto Currencies from "crypto_currencies.json" file
    Return list of coin dictionaries with symbol, name, and instrument details
    '''
    try:
        with open('crypto_currencies.json', 'r') as f:
            crypto_data = json.load(f)
        return crypto_data['coins']
    except FileNotFoundError:
        print('"crypto_currencies.json" not found.')
        return []
    except Exception as e:
        print(f"Error loading coins.json: {e}")
        return []


def get_historical_data(instrument, interval='day', from_ts=None, to_ts=None, api_key=API_KEY, market=MARKET, limit=LIMIT):
    
    """
    Fetch historical data from CoinDesk API
    interval: 'day' or 'hour'
    """

    if interval not in ['day', 'hour']:
        raise ValueError("Interval must be either 'day' or 'hour'")

    url = f'https://data-api.coindesk.com/index/cc/v1/historical/{interval}s'
    params = {
        'market': market,
        'instrument': instrument,
        'aggregate': 1,
        'fill': 'true',
        'apply_mapping': 'true',
        'response_format': 'JSON',
        'groups': 'OHLC, Volume, ID, Message',
        'limit': limit,
        'api_key': api_key
    }

    if to_ts:
        params['to_ts'] = int(to_ts)
    if from_ts:
        params['from_ts'] = int(from_ts)

    headers = {'Content-type': 'application/json; charset=UTF-8'}
    
    try:
        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"Error fetching data: {response.status_code} - {response.text}")

        return response.json()
    
    except Exception as e:
        print(f'Error occured for {instrument}: {e}')
        return None


def get_complete_historical_data(instrument, interval='day', api_key=API_KEY, market=MARKET):

    """
    Fetch all historical data by making multiple API calls using loop
    """
    all_records = []
    # batch_count = 0
    current_time = int(time.time())
    to_timestamp = current_time

    while True:
        # batch_count += 1
        # print(f"[{instrument}] Fetching batch #{batch_count} (to_ts={to_timestamp})")

        batch_data = get_historical_data(
            instrument=instrument,
            interval=interval,
            to_ts=to_timestamp,
            api_key=api_key,
            market=market,
            limit=LIMIT
        )

        if not batch_data or 'Data' not in batch_data:
            break

        batch_records = batch_data['Data']
        if not batch_records:
            break

        all_records.extend(batch_records)
        if len(batch_records) < LIMIT:
            break

        old_timestamp = min(entry.get('TIMESTAMP', current_time) for entry in batch_records)
        to_timestamp = old_timestamp - 1

        time.sleep(1)

    print()
    return {'Data': all_records}


# def convert_to_inr(usd_value):

#     '''
#     Converts currency from USD to INR
#     '''
#     c = CurrencyRates()
#     return c.convert('USD', 'INR', usd_value)


def json_to_csv(data, coin='BTC', interval='daily', market=MARKET):
    
    '''
    Process CoinDesk JSON and save required columns to CSV.
    Required output columns: Date, Open, High, Low, Close, Volume, Volume Quote, Market, Instrument, Total Trade
    All USD currency fields (Open, High, Low, Close, Volume Quote) are converted to INR (values replaced).
    Returns path to saved CSV or None on failure.
    '''

    if not data or 'Data' not in data:
        print('No data to process.')
        return None

    records = []
    #usd_to_inr = 88.27

    for entry in data['Data']:
        try:
            ts = entry.get('TIMESTAMP')
            date = pd.to_datetime(ts, unit='s') if ts else None

            record = {
                'Date': date,
                'Open': entry.get('OPEN', 0),
                'High': entry.get('HIGH', 0),
                'Low': entry.get('LOW', 0),
                'Close': entry.get('CLOSE', 0),
                'Volume': entry.get('VOLUME', 0),
                'Volume Quote': entry.get('QUOTE_VOLUME', 0),
                'Market': entry.get('MARKET', market),
                'Instrument': entry.get('INSTRUMENT', coin),
                'Total Index': entry.get('TOTAL_INDEX_UPDATES', None)
            }

            records.append(record)

        except Exception as e:
            print(f'Skipping record due to error: {e}.')

    df = pd.DataFrame(records)
    df.sort_values(by="Date", ascending=False, inplace=True)

    filename = f"{coin}_INR_{interval.capitalize()}_Historical_data.csv"
    try:
        df.to_csv(filename, index=False)
        print(f'Saved {filename} Successfully.')
        return filename
    except Exception as e:
        print(f'Error in saving CSV: {e}')
        return None
    

def fetch_crypto_currencies(interval='day'):

    '''
    Fetch and Process all Crypto Currencies from "crypto_currencies.json" file
    '''
    coins = load_crypto_currencies()
    print(f'\n\nStarting to process CryptoCurrencies for {interval}s data...\n')

    for coin in coins:
        try:
            symbol = coin['symbol']
            instrument = coin['instrument']

            print(f"Processing {symbol} ({instrument})...")

            data = get_complete_historical_data(instrument=instrument, interval=interval)

            if data and data.get('Data'):
                json_to_csv(data, coin=symbol, interval=interval)

        except Exception as e:
            print(f"Error occured for {coin}: {e}")

    print(f'\nEnding to process CryptoCurrencies for {interval}s data...')


if __name__ == '__main__':
    try:
        interval_choice = 'hour'
        fetch_crypto_currencies(interval=interval_choice)
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
    except Exception as e:
        print(f"Error: {e}")