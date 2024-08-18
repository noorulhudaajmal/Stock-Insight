# Stock Insight
Stock Insight is a Streamlit application designed to provide insights into stock data, including historical data, relationships among stock variables, and forecasts for future stock prices and trading volumes.

## Features
- **Historical Data Table:** View historical stock data with a detailed table.
- **Quick Summary Plot:** Get a quick overview of stock performance.
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

## Installation
To install the necessary packages, use the following commands:
```shell
pip install -r requirements.txt
```

## Usage
1. **Load Data:** The application initializes by loading stock data (default data) from a CSV file.

2. **Select Stock and Date Range:**
- Choose a stock ticker symbol from available list.
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