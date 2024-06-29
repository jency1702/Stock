import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

symbol = 'AAPL'
start_date = '2000-01-01'
end_date = '2020-01-01'

# Fetch historical stock data
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Calculate additional features
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
stock_data['RSI'] = 100 - (100 / (1 + stock_data['Close'].pct_change().rolling(window=14).mean() / stock_data['Close'].pct_change().rolling(window=14).std()))
stock_data['DayOfWeek'] = stock_data.index.dayofweek
stock_data['DayOfMonth'] = stock_data.index.day

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Extract features and target variable
features = stock_data[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'DayOfWeek', 'DayOfMonth']]
target = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Fetch historical stock data for the year 2024
stock_data_2024 = yf.download(symbol, start='2024-01-01', end='2024-12-31')

# Calculate additional features for 2024 data
stock_data_2024['SMA_20'] = stock_data_2024['Close'].rolling(window=20).mean()
stock_data_2024['SMA_50'] = stock_data_2024['Close'].rolling(window=50).mean()
stock_data_2024['EMA_20'] = stock_data_2024['Close'].ewm(span=20, adjust=False).mean()
stock_data_2024['RSI'] = 100 - (100 / (1 + stock_data_2024['Close'].pct_change().rolling(window=14).mean() / stock_data_2024['Close'].pct_change().rolling(window=14).std()))
stock_data_2024['DayOfWeek'] = stock_data_2024.index.dayofweek
stock_data_2024['DayOfMonth'] = stock_data_2024.index.day

# Drop rows with NaN values in 2024 data
stock_data_2024.dropna(inplace=True)

# Extract features for prediction
features_2024 = stock_data_2024[['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'DayOfWeek', 'DayOfMonth']]

# Use the trained model to make predictions
predictions_2024 = model.predict(features_2024)

# Display the predicted prices for 2024
df_predictions_2024 = pd.DataFrame({'Date': stock_data_2024.index, 'Predicted_Close': predictions_2024})
print(df_predictions_2024)

# Plot the predicted prices for 2024
plt.figure(figsize=(10, 6))
plt.plot(stock_data_2024.index, predictions_2024, label='Predicted Prices (2024)')
plt.title(f'{symbol} Predicted Stock Prices for 2024')
plt.xlabel('Date')
plt.ylabel('Predicted Closing Price (USD)')
plt.legend()
plt.savefig('plots/predicted_prices_2024.png')
plt.show()
