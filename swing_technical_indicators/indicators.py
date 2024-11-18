import pandas as pd
import  numpy as np

def calculate_rsi(data: pd.DataFrame, column_name: str = 'Closing_price', period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the price data.
    - column_name (str): The name of the column to calculate RSI on (default is 'Closing_price').
    - period (int): The number of periods to use for RSI calculation (default is 14).

    Returns:
    - pd.Series: A Pandas Series containing the RSI values.
    """
    
    # Ensure the column exists
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Step 1: Calculate the daily change in the specified column (e.g., Closing Price)
    delta = data[column_name].diff()
    
    # Step 2: Calculate Gain and Loss
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Step 3: Create a DataFrame for Gain and Loss
    gain_df = pd.DataFrame(gain, columns=['Gain'])
    loss_df = pd.DataFrame(loss, columns=['Loss'])
    
    # Step 4: Calculate the rolling average gain and loss
    avg_gain = gain_df['Gain'].rolling(window=period, min_periods=1).mean()
    avg_loss = loss_df['Loss'].rolling(window=period, min_periods=1).mean()
    
    # Step 5: Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Step 6: Calculate the RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Align the result with the input DataFrame
    rsi = pd.Series(rsi, name=f"RSI_{period}").fillna(0)
    
    return rsi

def calculate_200ema(data: pd.DataFrame, column_name: str = 'Closing_price', period: int = 200) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the price data.
    - column_name (str): The name of the column to calculate EMA on (default is 'Closing_price').
    - period (int): The number of periods to use for EMA calculation (default is 200).

    Returns:
    - pd.Series: A Pandas Series containing the EMA values.
    """
    
    # Ensure the column exists
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Calculate the EMA
    ema = data[column_name].ewm(span=period, adjust=False).mean()
    
    # Return the EMA as a Pandas Series
    return pd.Series(ema, name=f"EMA_{period}")

def calculate_swing_support_resistance(data: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
    """
    Calculate Support and Resistance levels for Swing Trading using Pivot Points for the given timeframe.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the price data.
    - timeframe (str): The timeframe for pivot point calculation, either 'daily', 'weekly', or 'monthly'.

    Returns:
    - pd.DataFrame: DataFrame with Pivot Points, Support, and Resistance levels, forward filled for non-resampled rows.
    """
    # Ensure the 'Date' column is in datetime format and set it as the index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    if timeframe == 'daily':
        high_col = 'Day_high'
        low_col = 'Day_low'
        close_col = 'Closing_price'
    elif timeframe == 'weekly':
        # Resample for weekly data
        resampled_data = data.resample('W').agg({
            'Day_high': 'max',
            'Day_low': 'min',
            'Closing_price': 'last'
        })
        high_col = 'Day_high'
        low_col = 'Day_low'
        close_col = 'Closing_price'
    elif timeframe == 'monthly':
        # Resample for monthly data
        resampled_data = data.resample('M').agg({
            'Day_high': 'max',
            'Day_low': 'min',
            'Closing_price': 'last'
        })
        high_col = 'Day_high'
        low_col = 'Day_low'
        close_col = 'Closing_price'
    else:
        raise ValueError("Timeframe must be 'daily', 'weekly', or 'monthly'.")

    # Calculate Pivot Point, Support, and Resistance levels
    resampled_data['Pivot_Point'] = (resampled_data[high_col] + resampled_data[low_col] + resampled_data[close_col]) / 3
    resampled_data['Support_1'] = (2 * resampled_data['Pivot_Point']) - resampled_data[high_col]
    resampled_data['Resistance_1'] = (2 * resampled_data['Pivot_Point']) - resampled_data[low_col]
    resampled_data['Support_2'] = resampled_data['Pivot_Point'] - (resampled_data[high_col] - resampled_data[low_col])
    resampled_data['Resistance_2'] = resampled_data['Pivot_Point'] + (resampled_data[high_col] - resampled_data[low_col])

    # Forward fill the calculated levels for the original data
    resampled_data = resampled_data[['Pivot_Point', 'Support_1', 'Resistance_1', 'Support_2', 'Resistance_2']]
    data = data.join(resampled_data, how='outer')
    
    # Fill NaN values using forward fill and backward fill
    data[['Pivot_Point', 'Support_1', 'Resistance_1', 'Support_2', 'Resistance_2']] = data[['Pivot_Point', 'Support_1', 'Resistance_1', 'Support_2', 'Resistance_2']].ffill().bfill()

    # Reset the index for the final output
    data.reset_index(inplace=True)

    return data

def calculate_50ema(data: pd.DataFrame, column_name: str = 'Closing_price', period: int = 50) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the price data.
    - column_name (str): The name of the column to calculate EMA on (default is 'Closing_price').
    - period (int): The number of periods to use for EMA calculation (default is 200).

    Returns:
    - pd.Series: A Pandas Series containing the EMA values.
    """
    
    # Ensure the column exists
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Calculate the EMA
    ema = data[column_name].ewm(span=period, adjust=False).mean()
    
    # Return the EMA as a Pandas Series
    return pd.Series(ema, name=f"EMA_{period}")

def calculate_100ema(data: pd.DataFrame, column_name: str = 'Closing_price', period: int = 100) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) for a given DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the price data.
    - column_name (str): The name of the column to calculate EMA on (default is 'Closing_price').
    - period (int): The number of periods to use for EMA calculation (default is 200).

    Returns:
    - pd.Series: A Pandas Series containing the EMA values.
    """
    
    # Ensure the column exists
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Calculate the EMA
    ema = data[column_name].ewm(span=period, adjust=False).mean()
    
    # Return the EMA as a Pandas Series
    return pd.Series(ema, name=f"EMA_{period}")

def calculate_bollinger_bands(data, period=20, num_std_dev=2):
    ##Period 20 is ideal for swing traders
    sma = data.rolling(window=period).mean()
    std_dev = data.rolling(window=period).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    
    bands =  pd.DataFrame({
        'Price': data,
        'SMA': sma,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })

    bands.fillna(method='ffill', inplace=True)  # Forward-fill
    bands.fillna(method='bfill', inplace=True)  # Backward-fill

    return bands

