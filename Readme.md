# Stock Insight
Stock Insight is a Streamlit application designed to provide insights into stock data for companies listed on the Pakistan Stock Exchange (PSX), including historical data, relationships among stock variables, and forecasts for future stock prices and trading volumes. It leverages LSTM (Long Short-Term Memory) networks for model training to forecast future stock and trading volumes.

## Features
- **Historical Data Table:** View historical stock data with a detailed table.
- **Quick Summary:** Get a quick overview of stock performance.
- **Stock Trend Over Time:** Analyze stock trends over a selected period.
- **Variable Relationships:** Explore relationships among different stock variables.
- **Stock Forecasting:** Predict future stock prices and trading volumes using pre-trained models.
- **Volume Forecasting:** Predict future trading volumes and visualize the results.

## Requirements
- Python 3.8+
- TensorFlow
- Pandas
- Streamlit
- Plotly
- psx-data-reader

## LSTM Model Training
The historical data of top performing PSX companies is used to train the neural network and the model is deployed to forecast future trends. The project aims at forecasting the future Volume(strength indicator of market) and
Open (the opening price of market) from the past trends.

## Installation
1. Clone the repo
```shell
git clone https://github.com/noorulhudaajmal/Stock-Insight.git
cd /Stock-Insight
```
2. To install the necessary packages, use the following commands:
```shell
pip install -r requirements.txt
```
3. Run the app:
```shell
streamlit run app.py
```

## Usage
1. **Load Data:** The application initializes by loading stock data (default data) from a CSV file.

2. **Select Stock and Date Range:**
   - Choose a stock ticker symbol from the available list.
   - Select a start and end date for the data range.

3. **View Historical Data:**
   - Display historical stock data in a table.
   - View a quick summary of stock performance.
   - Analyze Specific Dates:
     Select a specific date to view stock open, high, low, close prices, volume, and percentage change from the previous trading day.

4. **Explore Relationships Among Variables:**
   - Choose a variable to explore its relationship with other stock variables through interactive plots.

5. **Forecasting:**
   - Enter the number of days to forecast for both stock prices and trading volumes.
   - Visualize the forecast results with tables and line charts.
   - Compare actual vs. predicted values.

---
