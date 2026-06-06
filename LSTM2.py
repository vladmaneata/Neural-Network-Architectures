import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

def create_sequences(data, seq_length):
    """Creates input sequences and corresponding output values."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y).reshape(-1, 1)

def lstm_predict(X_train, y_train, X_test, hidden_dim, learning_rate, epochs):
    """Simple LSTM-like prediction using matrix operations."""
    seq_length = X_train.shape[1]
    input_dim = 1
    output_dim = 1

    U = np.random.randn(hidden_dim, seq_length)
    W = np.random.randn(hidden_dim, hidden_dim)
    V = np.random.randn(output_dim, hidden_dim)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for epoch in range(epochs):
        for i in range(len(X_train)):
            x, y = X_train[i], y_train[i]
            prev_s = np.zeros((hidden_dim, 1))
            for t in range(seq_length):
                x_t = x[t].reshape(1, 1)
                mulu = np.dot(U[:, t].reshape(hidden_dim, 1), x_t.T)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                prev_s = s
            pred = np.dot(V, s)
            error = pred - y
            dV = np.dot(error, s.T)
            dU = np.zeros_like(U)
            dW = np.zeros_like(W)

            for t in range(seq_length):
                x_t = x[t].reshape(1, 1)
                ds = np.dot(V.T, error) * s * (1 - s)
                dU[:, t] += np.dot(ds, x_t.T)[0,0] #corrected line
                dW += np.dot(ds, prev_s.T)
                prev_s = sigmoid(np.dot(U[:, t].reshape(hidden_dim, 1), x_t.T) + np.dot(W, prev_s))

            V -= learning_rate * dV
            U -= learning_rate * dU
            W -= learning_rate * dW

    predictions = []
    for x in X_test:
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(seq_length):
            x_t = x[t].reshape(1, 1)
            mulu = np.dot(U[:, t].reshape(hidden_dim, 1), x_t.T)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            prev_s = s
        predictions.append(np.dot(V, s)[0, 0])
    return np.array(predictions).reshape(-1, 1)

# Load training data from web
url_train = 'http://www.math.uaic.ro/~stoleriu/Google_Stock_Price_Train.csv'
s_train = requests.get(url_train).content
dataset_train = pd.read_csv(io.StringIO(s_train.decode('utf-8')))

print(dataset_train.shape)
print(dataset_train.head(7))
training_set = dataset_train.iloc[:, 1:2].values

# the MinMaxScaler library
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# reate a data structure with 60 timesteps and one output as an Array of x_train and y_train
seq_length = 60
X_train, y_train = create_sequences(training_set_scaled, seq_length)

# Load test data from web
url_test = 'http://www.math.uaic.ro/~stoleriu/Google_Stock_Price_Train.csv'
s_test = requests.get(url_test).content
dataset_test = pd.read_csv(io.StringIO(s_test.decode('utf-8')))
real_stock_price = dataset_test.iloc[:, 1:2].values

# the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - seq_length:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test, _ = create_sequences(inputs, seq_length)

# Simple LSTM-like prediction
hidden_dim = 50
learning_rate = 0.001
epochs = 100

predicted_stock_price = lstm_predict(X_train, y_train, X_test, hidden_dim, learning_rate, epochs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualize our data results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(f"RMSE: {rmse}")