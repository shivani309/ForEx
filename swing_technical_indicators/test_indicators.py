import pandas as pd
from swing_technical_indicators import calculate_rsi, calculate_ema, calculate_bollinger_bands, calculate_support_resistance

def test_rsi():
    data = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
    rsi = calculate_rsi(data)
    assert rsi.iloc[-1] is not None, "RSI calculation failed"

def test_ema():
    data = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
    ema = calculate_ema(data)
    assert ema.iloc[-1] is not None, "EMA calculation failed"

def test_bollinger_bands():
    data = pd.Series([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
    bands = calculate_bollinger_bands(data)
    assert not bands.isna().any().any(), "Bollinger Bands calculation failed"

def test_support_resistance():
    data = pd.DataFrame({
        'Day_high': [100, 105, 110],
        'Day_low': [95, 97, 99],
        'Closing_price': [98, 102, 107]
    })
    sr = calculate_support_resistance(data)
    assert not sr.isna().any().any(), "Support and Resistance calculation failed"
