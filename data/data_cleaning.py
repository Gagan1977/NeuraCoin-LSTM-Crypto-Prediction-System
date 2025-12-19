import pandas as pd

coins = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOGE', 'SOL']

print('No. of zeros data in Cryptocurrencies Hourly data:')
for coin in coins:
    df = pd.read_csv(f'hourly_data/{coin}_INR_Hour_Historical_data.csv')
    
    zero_rows = df[df['Volume'] == 0]
    print(f'{coin}:', len(zero_rows))

print()

print('No. of zeros data in Cryptocurrencies Daily data:')
for coin in coins:
    df = pd.read_csv(f'daily_data/{coin}_INR_Day_Historical_data.csv')

    zero_rows = df[df['Volume'] == 0]
    print(f'{coin}:', len(zero_rows))