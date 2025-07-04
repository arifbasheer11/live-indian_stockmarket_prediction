# Live Indian Stock Market Prediction

This project is a web application that predicts the movement of Indian stock prices using AI-powered LSTM neural networks. It provides live price updates, technical analysis, and investment recommendations for a wide range of NSE-listed stocks.

## Features

- **Live Stock Prediction:** Select from a list of Indian stocks and get AI-based predictions for the next movement.
- **LSTM Neural Network:** Uses a Long Short-Term Memory (LSTM) model trained on recent price history.
- **Technical Analysis:** Displays moving averages, RSI, volume change, and more.
- **Investment Recommendation:** Offers a recommendation (Buy/Hold/Sell) with reasons and financial fundamentals.
- **Modern UI:** Clean, responsive interface built with Bootstrap.

## How It Works

1. **User selects a stock** from the dropdown menu on the homepage.
2. The app fetches recent historical data using `yfinance`.
3. Data is preprocessed and fed into an LSTM model for prediction.
4. The result page displays:
   - Predicted movement (Up/Down) and confidence
   - Live price and change
   - Technical indicators and advanced analysis
   - Investment recommendation and financials
   - Price chart

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd live-indian_stockmarket_prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies:
  ```
  numpy
  scipy
  pandas
  scikit-learn
  gunicorn
  flask
  tensorflow
  yfinance
  matplotlib
  ```

## Usage

- Open your browser and go to `http://127.0.0.1:5000/`.
- Select a stock and click "Predict Movement".
- View the prediction, technical analysis, and recommendations.

## Notes

- **Disclaimer:** Stock predictions are inherently uncertain. Always conduct your own research before making investment decisions.
- The model uses the last 60 data points for prediction and is retrained on each request for up-to-date results.

## File Structure

- `app.py` — Main Flask application and ML logic
- `templates/index.html` — Homepage (stock selection)
- `templates/result.html` — Prediction result and analysis page
- `requirements.txt` — Python dependencies 
