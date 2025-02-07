# User selects a currency pair
import joblib
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import swing_technical_indicators 
from statsmodels.tsa.seasonal import seasonal_decompose

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from phi.utils.pprint import pprint_run_response
from phi.agent import Agent , RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai 
import os 

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/getPrediction": {"origins": "http://localhost:4200"}})
# api_key = 'EWTP9LXAZRDLRHB8'
api_key = '89PMD611NPUTS89W'

# Function to fetch Forex data
def fetch_forex_data(from_currency, to_currency):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'FX_DAILY',
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'apikey': api_key,
        'outputsize': 'compact'
    }
    response = requests.get(url, params=params)
    data = response.json()

    time_series = data.get('Time Series FX (Daily)', {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df.rename(columns={
        "1. open": "Open_price",
        "2. high": "Day_high",
        "3. low": "Day_low",
        "4. close": "Closing_price"
    }, inplace=True)

    df['Currency Pair'] = f'{from_currency}/{to_currency}'

    # Filter past 60 days to ensure enough data for sequences
    last_100_days = datetime.now() - timedelta(days=100)
    df = df[df.index >= last_100_days]

    return df

def deseasonalize_column(data, column, period=7):
    # Decompose the column
    decomposition = seasonal_decompose(data[column], model='additive', period=period, extrapolate_trend='freq')
    
    # Extract the residual component as deseasonalized data
    deseasonalized = data[column] - decomposition.seasonal
    
    return deseasonalized

# Function to create sequences for CNN
def create_sequences(X, y, sequence_length, forecast_horizon=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length - forecast_horizon + 1):  
        X_seq.append(X[i:i + sequence_length])  
        y_seq.append(y[i + sequence_length : i + sequence_length + forecast_horizon])  # Next 7 values
    return np.array(X_seq), np.array(y_seq)

def recommendationByAgenticRAG(currencyPair):
    os.environ['GROQ_API_KEY'] = "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"
    print(os.getenv("GROQ_API_KEY"))

    model = Groq(
    id="llama3-groq-70b-8192-tool-use-preview",
    api_key="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"
    )
    print("Authentication successful!")

    #openai.api_key = os.getenv('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')

    ## Web search agent
    web_search_agent = Agent(
        name = 'web_search_agent',
        role = 'search the web for information',
        model = Groq(id = 'llama-3.2-3b-preview',api_key = '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'),
        tools = [DuckDuckGo()],
        instructions = ['Always include sources and numbers in actual to support your reasoning'],
        show_tool_calls = False,
        markdown = True

    )

    ## Finance agent 

    finance_agent = Agent(
        name = 'finance_agent',
        model = Groq(id = 'llama-3.2-3b-preview'),
        tools=[YFinanceTools(stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            technical_indicators=True)],
        show_tool_calls=False,
        description="You are an advanced financial analyst providing stock insights with precise technical and fundamental analysis.",
        instructions=["Format your response using markdown and use tables to display data where possible.",
            "Provide a Buy/Sell recommendation including the following details:",
            "- Target Price",
            "- Stop-Loss",
            "- Holding Period",
            "- Expected Average Profit %",
            "- Key Fundamental Indicators like P/E Ratio, EPS, Dividend Yield, and Market Cap.",
            "No need to include company history such as founding details."],
        markdown=True
    )

    multi_ai_agent = Agent(
        team = [web_search_agent,finance_agent],
        model = Groq(id = 'llama-3.2-3b-preview'),
        instructions = ["Always include sources and numbers to support reasoning.",
                        "Do not forget to begin recommendations with the name of the stock",
            "Format response in markdown with tables for data representation.",
            "Ensure all technical indicators are displayed and analyzed.",
            "Ensure that the recommendation whether to buy or sell is displayed along with target , stoploss and holding period are displayed at any cost in tabular format only."
            ],
        show_tool_calls= False,
        markdown=True

    )

    formattedPair = currencyPair.replace("-", "") + "=X"
    # Run agent and return the response as a variable
    response = multi_ai_agent.run(f"Summarize your response and give recommendation over the stock JSWSTEEL")
    # Print the response in markdown format
    # pprint_run_response(response, markdown=True)

    if isinstance(response, RunResponse):
        recommendation_text = response.content  # Extracts only the text part
    else:
        recommendation_text = str(response)  # Fallback in case it's another format

    return recommendation_text


def calculate_technical_indicators(data):
    data['EMA_100'] = swing_technical_indicators.calculate_100ema(data, 'Closing_price', 100)
    data['EMA_200'] = swing_technical_indicators.calculate_200ema(data, 'Closing_price', 200)
    data['EMA_50'] = swing_technical_indicators.calculate_100ema(data, 'Closing_price', 50)
    # data['RSI'] = swing_technical_indicators.calculate_rsi(data,'Closing_price',21)
    macd = swing_technical_indicators.calculate_macd(data, column_name='Closing_price', short_period=12, long_period=26, signal_period=9)
    for i in macd.columns:
        data[i] = macd[i]
        
    bollinger_data = swing_technical_indicators.calculate_bollinger_bands(data['Closing_price'])
    data = data.join(bollinger_data)
    data.drop('Price',axis=1, inplace=True)

    return data

def make_recommendation(data):
    numeric_cols = ['Closing_price', 'EMA_50', 'EMA_200', 'MACD', 'Lower Band', 'Upper Band']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Extract the last week's (7 days) data
    last_week_data = data.tail(7)
    
    # Calculate the latest values for the indicators based on the last week's data
    latest = last_week_data.iloc[-1]  # Latest data point in the last 7 days
    
    # Calculate the trend using the 50 EMA and 200 EMA
    trend = "neutral"  # Default to neutral
    if latest['Closing_price'] > latest['EMA_50'] and latest['Closing_price'] > latest['EMA_200']:
        trend = "bullish"
        print(trend,' ema')
        print(f"The value of EMA 50 : {latest['EMA_50']} and 200 EMA : {latest['EMA_200']} suggest bullish trend")
        
    elif latest['Closing_price'] < latest['EMA_50'] and latest['Closing_price'] < latest['EMA_200']:
        trend = "bearish"
        print(f"The value of EMA 50 : {latest['EMA_50']} and 200 EMA : {latest['EMA_200']} suggest bearish trend")
    #print(trend)
    # Check RSI for the last 7 days
    """rsi_signal = "neutral"
    if latest['RSI'] < 30:
        rsi_signal = "bullish"
        #print(f"RSI values: {latest['RSI']} , rsi_signal:{rsi_signal}")
    elif latest['RSI'] > 70:
        rsi_signal = "bearish"""
        #print(f"RSI values: {latest['RSI']} , rsi_signal:{rsi_signal}")


    # Check MACD (and MACD signal line) for the last 7 days
    macd_signal = "neutral"
    if latest['MACD'] > 0:
        macd_signal = "bullish"
        print(f"MACD values: {latest['MACD']} , macd: {macd_signal}")
    elif latest['MACD'] < 0:
        macd_signal = "bearish"
        print(f"MACD values: {latest['MACD']} , macd: {macd_signal}")

    # Check Bollinger Bands for the last 7 days
    bb_signal = "neutral"
    if latest['Closing_price'] < latest['Lower Band']:
        bb_signal = "bullish"
        print(f"Bollinger Bands : lower band suggests : {bb_signal}")
    elif latest['Closing_price'] > latest['Upper Band']:
        bb_signal = "bearish"
        print(f"Bollinger Bands : upper band suggests : {bb_signal}")

    # Combine all indicators for decision
      # Default decision is to hold
    target = None
    stop_loss = None
    #print(latest['Closing_price'])
    # Decision making based on the trend, RSI, MACD, and Bollinger Bands
    #print(trend , '1')
    if trend == "bullish" and macd_signal == "bullish":
        #print('yes')
        decision = "BUY"
        target = latest['Closing_price'] * 1.05  # Set target at 5% higher
        stop_loss = latest['Closing_price'] * 0.97  # Set stop loss at 3% lower

        
    elif trend == "bearish"  and macd_signal == "bearish":
        #print('no')
        decision = "SELL"
        target = latest['Closing_price'] * 0.95  # Set target at 5% lower
        stop_loss = latest['Closing_price'] * 1.03  # Set stop loss at 3% higher
    else :
        decision = "WAIT"

    #print(trend)
    #print(decision, target, stop_loss)
    return decision, target, stop_loss
    

def trading_pipeline(symbol_from, symbol_to, api_key, days=200):
    # Fetch forex data
    data = fetch_forex_data(symbol_from, symbol_to)
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Make recommendation
    decision, target, stop_loss = make_recommendation(data)
    print('Currency pair : ',symbol_from, symbol_to)
    # Print recommendation and details
    print(f"Recommendation: {decision}")
    if decision != "WAIT":
        print(f"Target Price: {target:.4f}")
        print(f"Stop Loss: {stop_loss:.4f}")
        print(f"- Target/Stop Loss set at 5% and 3% based on the current price.")
        print("Recommendation Based on the Following Justifications:")
    
    elif decision == "WAIT":
        print(f"- Hold for 1 week to 1 month.")
    else:
        print(f"- Set a stop loss at {stop_loss:.4f} and target price at {target:.4f}.")
    print(f"- Trend: {decision} trend confirmed by indicators.")

    message = f"Currency pair: {symbol_from}/{symbol_to}\n"
    message += f"Recommendation: {decision}\n"
    
    if decision != "WAIT":
        message += f"Target Price: {target:.4f}\n"
        message += f"Stop Loss: {stop_loss:.4f}\n"
        message += "- Target/Stop Loss set at 5% and 3% based on the current price.\n"
        message += "Recommendation Based on the Following Justifications:\n"
    
    elif decision == "WAIT":
        message += "- Hold for 1 week to 1 month.\n"
    
    else:
        message += f"- Set a stop loss at {stop_loss:.4f} and target price at {target:.4f}.\n"
    
    message += f"- Trend: {decision} trend confirmed by indicators.\n"

    return message


def recommendationByTechIndi(symbol_from, symbol_to):
    return trading_pipeline(symbol_from, symbol_to, api_key, days=200)
    

@app.route('/getPrediction', methods=['POST'])
def predict_result():
    data = request.get_json()
    print("data recived---->", data)

    from_currency, to_currency = data['currencyPair'].split("-")
    df = fetch_forex_data(from_currency, to_currency)
    
    df["Open_price"] = pd.to_numeric(df["Open_price"], errors='coerce')
    df["Day_high"] = pd.to_numeric(df["Day_high"], errors='coerce')
    df["Day_low"] = pd.to_numeric(df["Day_low"], errors='coerce')
    df["Closing_price"] = pd.to_numeric(df["Closing_price"], errors='coerce')

    # Identify non-stationary columns
    non_stationary_columns = ['Day_high', 'Day_low', 'Open_price']

    # Create deseasonalized columns
    for col in non_stationary_columns:
        deseasonalized_col = deseasonalize_column(df, col, period=14)  # Adjust the period as needed
        df[f'Deseasonalized_{col}'] = deseasonalized_col

    df['EMA_100'] = swing_technical_indicators.calculate_100ema(df, 'Closing_price', 100)
    df['EMA_200'] = swing_technical_indicators.calculate_200ema(df, 'Closing_price', 200)
    df['EMA_50'] = swing_technical_indicators.calculate_100ema(df, 'Closing_price', 50)
    #df['RSI'] = swing_technical_indicators.calculate_rsi(df,'Closing_price',21)

    data_macd = swing_technical_indicators.calculate_macd(df, column_name='Closing_price', short_period=12, long_period=26, signal_period=9)

    for i in data_macd.columns:
        df[f"{i}"] = data_macd[i]

    bollinger_data = swing_technical_indicators.calculate_bollinger_bands(df['Closing_price'])
    y = df["Closing_price"]
    df = df.join(bollinger_data)
    df.drop('Price',axis=1, inplace=True)
    df.drop('Closing_price',axis=1, inplace=True)
    df.drop('Currency Pair',axis=1, inplace=True)

    scaler_X = None
    scaler_y = None
    # Load saved scalers and models
    if(data['currencyPair'] == "USD-INR"):
        scaler_X = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/USDINR/DL/usd_inr_scaler_X')
        scaler_y = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/USDINR/DL/usd_inr_scaler_y')
    if(data['currencyPair'] == "EUR-INR"):
        scaler_X = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/EURINR/eurinrscalerx')
        scaler_y = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/EURINR/eurinrscalery')
    if(data['currencyPair'] == "JPY-INR"):
        scaler_X = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/JPYINR/jpyinr_scalerx')
        scaler_y = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/JPYINR/jpyinr_scalery')
    if(data['currencyPair'] == "GBP-INR"):
        scaler_X = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/GBPINR/gbpinr_scalerx')
        scaler_y = joblib.load(r'C:/Users/uzmap/Documents/GitHub/ForEx/GBPINR/gbpinr_scalery')

    # Normalize data
    df_scaled = scaler_X.transform(df)

    # Create sequences
    X_seq, y_seq = create_sequences(df_scaled,y,sequence_length = 30,forecast_horizon=7)

    sequence_length = 30
    feature_count = X_seq.shape[2]  # Number of features

    # Define CNN Model
    cnn_model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(sequence_length, feature_count)),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(7)  # Output layer for predicting next 7 days
    ])

     # Load saved scalers and models
    if(data['currencyPair'] == "USD-INR"):
        cnn_model.load_weights(r"C:/Users/uzmap/Documents/GitHub/ForEx/USDINR/DL/usd_inr_cnn.h5")
    if(data['currencyPair'] == "EUR-INR"):
        cnn_model.load_weights(r"C:/Users/uzmap/Documents/GitHub/ForEx/EURINR/eurinrmodel.h5")
    if(data['currencyPair'] == "JPY-INR"):
        cnn_model.load_weights(r"C:/Users/uzmap/Documents/GitHub/ForEx/JPYINR/jpyinr_cnn_model.h5")
    if(data['currencyPair'] == "GBP-INR"):
        cnn_model.load_weights(r"C:/Users/uzmap/Documents/GitHub/ForEx/GBPINR/gbpinr_cnn_model.h5")
    
    # Predict next 7 days
    y_pred_scaled = cnn_model.predict(X_seq[-1].reshape(1, sequence_length, feature_count))

    # Convert predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    predicted_prices = []
    # Print results
    print("\nPredicted Closing Prices for Next 7 Days:")
    for i, price in enumerate(y_pred.flatten(), 1):
        predicted_prices.append({"day": i, "price": f"{price:.4f}"})

    print(predicted_prices)
    predicted_prices = [{"day": i, "price": float(price)} for i, price in enumerate(y_pred.flatten(), 1)]

    # recommendationMade = recommendationByAgenticRAG(data['currencyPair'])
    recommendationMade = recommendationByTechIndi(from_currency, to_currency)

    # Plot results
    # plt.figure(figsize=(10, 6))
    # plt.plot( y_pred.flatten(), marker='o', linestyle="--", color="red", label="Predicted Price")
    # plt.xlabel("Days Ahead")
    # plt.ylabel("Closing Price")
    # plt.title(f"Next 7 Days Prediction for {data['currencyPair']}")
    # plt.legend()
    # plt.grid()
    # plt.show()

    # Send the image data to the frontend as part of the response
    response_message = {
        "predictedPrices": predicted_prices,
        "recommendation": recommendationMade
    }

    return jsonify(response_message), 200

if __name__ == '__main__':
    app.run(debug=True)