📈 Indian Stock Movement Predictor



This Flask web application predicts the next movement (Up or Down) of Indian stocks using LSTM deep learning models and technical indicators. It also provides live stock data, advanced analysis, and interactive charts for better decision-making.

🚀 Features
🔍 Search from 400+ Indian NSE stocks (e.g., RELIANCE.NS, TCS.NS)

📊 Predict stock movement using a trained LSTM model

📉 Visualize the last 60 days' prices with interactive charts

💡 Advanced analysis:

20/50/200-day Moving Averages

RSI (Relative Strength Index)

Volume spike detection

Buy/Sell recommendations

🏷️ Real-time prices, company info, and sector

📈 Confidence score for predictions

🧠 Caching for faster performance

🛠️ Tech Stack
Backend: Flask, TensorFlow/Keras, Scikit-learn

Frontend: Jinja2 (HTML Templates), Chart.js (via JSON)

Data: yfinance for real-time stock data

Model: LSTM Neural Network (Sequential)

🧠 How It Works
User selects a stock and submits the form.

The app fetches the past 60+ days of closing prices.

An LSTM model is trained on this data to predict the next price.

The prediction is compared to the current price to determine:

Direction: Up or Down

Confidence: % difference

A recommendation is generated based on:

RSI

Volume trends

Moving averages

🔐 Notes
Uses lru_cache and TTL hash for caching stock data (15 mins).

Trains the LSTM model in real-time per request (optional: cache predictions).

Tickers are from NSE and must end with .NS (e.g., INFY.NS)
