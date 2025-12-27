# NeuraCoin: LSTM Cryptocurrency Prediction System

NeuraCoin is an end-to-end deep learning framework designed to forecast cryptocurrency prices with high precision. Built on PyTorch, it utilizes Long Short-Term Memory (LSTM) networks to capture temporal dependencies in volatile market data.

The system features a complete pipeline for real-time data fetching (via Binance API), data preprocessing, model training, evaluation, and future forecasting.

---

## Table of Contents
- Project Overview  
- Key Features  
- Technology Stack  
- Repository Structure  
- Installation and Setup  
- Usage Pipeline  
- Future Roadmap  
- License  

---

## Project Overview

This project provides a modular architecture for predicting cryptocurrency price movements. It treats price forecasting as a time-series regression problem, leveraging historical Open, High, Low, Close, and Volume (OHLCV) data. The system is designed to be scalable, supporting multiple cryptocurrencies and timeframes (daily and hourly).

---

## Key Features

- **Deep Learning Architecture**  
  Implements a custom LSTM network optimized for time-series data to prevent vanishing gradient problems.

- **Multi-Asset Support**  
  Capable of training on major cryptocurrencies including Bitcoin (BTC), Ethereum (ETH), Solana (SOL), Ripple (XRP), Litecoin (LTC), Dogecoin (DOGE), Cardano (ADA), and Bitcoin Cash (BCH).

- **Dual Timeframe Analysis**  
  Supports both Daily (long-term trend) and Hourly (short-term volatility) prediction models.

- **Automated Data Pipeline**  
  Includes scripts for fetching live data from the Binance API, cleaning dataset anomalies, and normalizing values.

- **Performance Metrics**  
  Evaluates models using industry-standard metrics:  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - Mean Absolute Percentage Error (MAPE)

- **Visualization**  
  Generates comparative plots of Predicted vs. Actual prices to visually assess model performance.

---

## Technology Stack

- **Language:** Python 3.9+  
- **Deep Learning:** PyTorch (torch, torch.nn)  
- **Data Manipulation:** Pandas, NumPy  
- **Data Scaling:** Scikit-learn (MinMaxScaler)  
- **Data Source:** Binance API (python-binance)  
- **Visualization:** Matplotlib, Seaborn  

---

## Repository Structure

```plaintext
NeuraCoin-System/
├── data/
│   ├── daily_data/              # CSV storage for daily candles
│   ├── hourly_data/             # CSV storage for hourly candles
│   ├── data_collection.py       # Script to fetch live data from CoinDesk API
│   ├── data_cleaning.py         # Utilities for data validation and sorting
│   └── crypto_currencies.json   # Configuration for supported coins
├── model/
│   ├── architectures/           # Neural network model definitions
│   │   ├── base_model.py
│   │   └── lstm.py
│   ├── checkpoints/             # Saved .pth model weights
│   ├── scalers/                 # Saved .pkl scalers for denormalization
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Testing and metrics calculation
│   ├── predict.py               # Inference script for future dates
│   └── preprocess.py            # Data normalization and sequence generation
├── results/                     # Output graphs and evaluation logs
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Installation and Setup

### Prerequisites
* Python 3.8 or higher
* pip package manager
* Virtual environment tool (optional but recommended)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/neuracoin-lstm.git](https://github.com/yourusername/neuracoin-lstm.git)
    cd neuracoin-lstm
    ```

2.  **Create and Activate Virtual Environment**
    ```bash
    python -m venv venv
    # Linux/MacOS:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```


---

## Usage Pipeline

The system is designed to be executed in a sequential pipeline.

### 1. Data collection

Fetch the latest historical market data from the Binance API.

```bash
python data/data_collection.py
```

Output: Raw CSV files are saved to `data/daily_data/` and `data/hourly_data/`.

### 2. Preprocessing

Clean, sort, and normalize the raw data. This step generates the training tensors and saves the scalers required for later denormalization.

```bash
python model/preprocess.py
```

Output: .pt files in `data/processed/` and .pkl files in `model/scalers/`.

### 3. Model Training

Train the LSTM model. The script automatically detects and utilizes a GPU (CUDA) if available.

```bash
python 4. model/train.py
```

Output: Best performing model weights are saved to `model/checkpoints/`.

### 5. Evaluation

Evaluate the trained model against a held-out test set. This generates accuracy metrics and visual plots.

```bash
python model/evaluate.py
```

Output: Performance graphs are saved to `results/`.

### 6. Prediction

Generate forecasts for future dates using the trained model.

```bash
# Example: Predict Bitcoin prices for the next 7 days
python model/predict.py --coin bitcoin --timeframe daily --days 7
```

Output: Prints the predicted Open, High, Low, Close, and Volume values to the console.


---

## Future Roadmap

The current version operates as a Command Line Interface (CLI) tool. The next phase of development (v2.0) aims to transform this into a full-stack web application.

* **Backend Migration**: Transition inference logic to a Django REST Framework API.
* **Asynchronous Processing**: Implement Celery and Redis to handle background model training and scheduled data ingestion.
* **Frontend Interface**: Develop a web-based dashboard for visualizing real-time predictions and model accuracy.
* **User Management**: Add authentication for users to save preferences and set price alerts.
* **Deployment**: Containerize the application using Docker for consistent deployment environments.

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
