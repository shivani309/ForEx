# Swing Technical Indicators

## Overview

`swing_technical_indicators` is a Python library designed for swing traders and technical analysts to easily compute popular technical indicators. This library is specifically tailored for trading strategies, providing various indicators like EMA (50, 100, 200), RSI, MACD, Bollinger Bands, and Swing Trading-based Support and Resistance levels.

## Features

This library provides the following technical indicators:

1. **Exponential Moving Average (EMA)**
   - **50 EMA**: Short-term trend indicator, commonly used to identify quick trend changes.
   - **100 EMA**: Medium-term trend indicator for capturing more stable trends.
   - **200 EMA**: Long-term trend indicator to determine the overall direction of the market.

2. **Relative Strength Index (RSI)**
   - Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.

3. **Moving Average Convergence Divergence (MACD)**
   - A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

4. **Bollinger Bands**
   - A volatility indicator that provides a relative definition of high and low prices, using a Simple Moving Average (SMA) and standard deviations.

5. **Support and Resistance Levels (Swing Trading)**
   - Calculated based on swing trading techniques, helping traders identify potential reversal points.

## Installation

To install the library locally, use:

```bash
pip install -e .

##  Usage
Here's a quick guide on how to use the library:

Importing the Library
import pandas as pd
import swing_technical_indicators as sti

data = pd.Series([61.390, 61.500, 61.530, 61.508, 61.368, 61.045, 60.987])

ema_50 = sti.calculate_ema(data, period=50)
print(ema_50)  # calculate ema 

rsi = sti.calculate_rsi(data, period=14)
print(rsi)
 
macd = sti.calculate_macd(data)
print(macd)

result_data = calculate_swing_support_resistance(data, timeframe='weekly')
print(result_data[['Date', 'Pivot_Point', 'Support_1', 'Resistance_1', 'Support_2', 'Resistance_2']])

bollinger_bands = sti.calculate_bollinger_bands(data, period=20)
print(bollinger_bands)



### Notes:

1. **Installation Section**: You can adjust the installation instructions based on your use case.
2. **Usage Examples**: This `README.md` provides clear usage examples, making it easier for others to understand how to implement your library.
3. **File Details**: Feel free to modify and add any additional features or sections as per your requirement.

This `README.md` file will serve as an excellent documentation resource for anyone using your library. Let me know if you need more help!



