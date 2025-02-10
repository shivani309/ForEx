# ForEx Forecasting Framework

## Introduction 

This project aims to improve Forex forecasting by addressing limitations in existing methodologies. By leveraging a dataset spanning the last 10 years, we capture long-term market trends, focusing specifically on the four major currency pairs used by Indian traders: USD/INR, EUR/INR, GBP/INR, and JPY/INR. The framework is tailored for swing trading, supporting positions held over days to weeks. A key contribution is the creation of the Swing Technical Indicators Library, designed to aid in market analysis and provide actionable insights.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technical Indicators Used](#technical-indicators-used)
6. [Alpha Vantage API Integration](#alpha-vantage-api-integration)
7. [Project Workflow](#project-workflow)
8. [Resources for Learning Forex Trading](#resources-for-learning-forex-trading)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

The Forex Currency Price Forecasting and Recommendation System is designed to analyze historical and real-time currency exchange rates, calculate technical indicators, and predict future price movements. The project is suitable for both beginner and experienced Forex traders.

---

## Features

- Forecasting Forex currency prices using advanced machine learning models.
- Analysis of historical data using various technical indicators.
- Personalized trading recommendations.
- Integration with Alpha Vantage API for real-time Forex data.
- Modular and extensible codebase for further enhancements.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shivani309/ForEx.git
   cd ForEx
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the Alpha Vantage API key (explained in the [Alpha Vantage API Integration](#alpha-vantage-api-integration) section).

---

## Usage

1. Prepare the dataset:
   - Ensure that you have historical Forex data in `.csv` format.
   - Place your dataset in the `data/` directory.

2. Run the main script:
   ```bash
   python main.py
   ```

3. View results and recommendations in the `results/` directory.

## Technical Indicators Used

This project calculates several technical indicators for data analysis and feature engineering, including but not limited to:

- Moving Average (MA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)

You can customize or add more indicators in the `indicators.py` module.

---

## Alpha Vantage API Integration

This project uses the [Alpha Vantage API](https://www.alphavantage.co/) to fetch real-time Forex data. To set up the API:

1. Sign up on the [Alpha Vantage website](https://www.alphavantage.co/) and get your free API key.
2. Add the API key to your environment variables or update it directly in the `config.py` file:
   ```python
   ALPHA_VANTAGE_API_KEY = 'your_api_key_here'
   ```
3. Use the `alpha_vantage` Python library (already included in `requirements.txt`) to interact with the API. Example usage:
   ```python
   from alpha_vantage.foreignexchange import ForeignExchange

   fx = ForeignExchange(key='your_api_key_here')
   data, _ = fx.get_currency_exchange_rate(from_currency='USD', to_currency='EUR')
   print(data)
   ```

For more details, refer to the official [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/).

---

## Project Workflow

1. **Data Collection:**
   - Download historical Forex data from reliable sources or use the Alpha Vantage API.
2. **Data Preprocessing:**
   - Handle missing values, normalize data, and calculate technical indicators.
3. **Model Development:**
   - Train and evaluate machine learning models for price prediction.
4. **Recommendation System:**
   - Generate buy/sell signals based on model predictions and indicator thresholds.
5. **Visualization:**
   - Plot trends, predictions, and recommendations for better insights.

---

## Resources for Learning Forex Trading

### Forex Basics
- [Babypips School of Pipsology](https://www.babypips.com/learn/forex)
- [Forex Trading for Beginners - Investopedia](https://www.investopedia.com/terms/f/forex.asp)

### Technical Indicators
- [Technical Indicators - TradingView](https://www.tradingview.com/)
- [MACD, RSI, and Other Indicators - Investopedia](https://www.investopedia.com/terms/t/technicalindicator.asp)

### Data and APIs
- [Alpha Vantage API](https://www.alphavantage.co/)

---

## Contributing

We welcome contributions! Please submit a pull request or open an issue for discussions.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

---

## License

**Copyright (C) 2025 Shivani309**

This software is licensed under the **Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. This means:
- You are free to use, modify, and distribute the Swing Technical Indicator Library **for non-commercial purposes**.
- You must **cite** this repository in any publication, project, or derivative work.
- For **commercial use**, please contact the author for licensing.

**Attribution Requirement:**
```
@software{shivani_swing_indicator,
  author = {Shivani309},
  title = {Swing Technical Indicator Library},
  year = {2025},
  url = {https://github.com/shivani309/ForEx}
}
```

For more details, refer to the [LICENSE](LICENSE) file.

