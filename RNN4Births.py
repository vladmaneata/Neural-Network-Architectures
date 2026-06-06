# a SimpleRNN with a very short "memory" and a fairly small architecture
# to predict a very "noisy" dataset (daily births)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Read data
df = pd.read_excel('Births.xlsx', header=0)
data = df['Births'].values.reshape(-1, 1).astype('float32')

# Scale data to range [0, 1] to help the RNN converge faster and avoid gradient issues
scaler = MinMaxScaler(feature_range=(0, 1))   # Normalize the data
scaled_data = scaler.fit_transform(data)

look_back = 7           # the model sees 'look_back' days to predict the next 1
# Create dataset with look-back-days lag
def create_dataset(dataset, look_back):   # Create dataset with look-back-days lag
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: (i + look_back), 0])   # a window of look_back days
        Y.append(dataset[i + look_back, 0])    # Y is the very next day after the swindow
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_data, look_back)

# Reshape X to match the Keras RNN layers requirements
X = np.reshape(X, (X.shape[0], look_back, 1))

# Split into 70% Train and 30% Test
train_size = int(len(X) * 0.7)
trainX, testX = X[0:train_size], X[train_size:]
trainY, testY = Y[0:train_size], Y[train_size:]

# Build the RNN model
model = Sequential([
    SimpleRNN(100, activation='relu', input_shape=(look_back, 1)),
    Dropout(0.4),
    Dense(1)   # final output layer, that produces a single predicted value
])

# Early stopping monitor
early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,                 # wait 30 epochs for an improvement before quitting
    verbose=1,                   # print a message when training stops
    restore_best_weights=True    # Keep the weights from the "best" epoch
)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    trainX, trainY,
    epochs=400,
    batch_size=32,
    validation_data=(testX, testY),
    #callbacks=[early_stop],      # you can add/remove the callback here
    verbose=0                     # no message is printed
)

# Make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Invert predictions back to original scale and round
train_predict = scaler.inverse_transform(train_predict)
train_predict = np.rint(train_predict)             # integer values
test_predict = scaler.inverse_transform(test_predict)
test_predict = np.rint(test_predict)               # integer values
actual_values = scaler.inverse_transform(Y.reshape(-1, 1))

# Create empty arrays for plotting
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[look_back: len(train_predict) + look_back, :] = train_predict

test_plot = np.empty_like(data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + look_back: len(data), :] = test_predict

last_sequence = scaled_data[-look_back:]  # Start with the very last look_back days of known data
current_batch = last_sequence.reshape((1, look_back, 1))

future_predictions = []

for i in range(14):
    current_pred = model.predict(current_batch, verbose=0)[0]
    future_predictions.append(current_pred)   # add the new prediction
    new_seq = np.append(current_batch[0, 1:, 0], current_pred)
    current_batch = new_seq.reshape((1, look_back, 1))

#  Invert scaling and round
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_predictions = np.rint(future_predictions)          # integer values

last_day_index = len(df) - 1     # X-axis indices
future_days = np.arange(last_day_index + 1, last_day_index + 15)

plt.figure(figsize=(12, 6))
plt.plot(df['Births'].values, '*:', label='Original data', color='green', alpha=0.3)   # original data

plt.plot(train_plot, label='Train prediction', color='blue')   # Training predictions
plt.plot(test_plot, label='Test prediction (lag 7)', color='orange')   # Testing predictions

plt.plot(future_days, future_predictions, '-', label='14-day forecast',
         color='red', linewidth=2, markersize=8)     # 14-Day forecast

plt.title('Daily births: historical vs. RNN prediction + 14-day forecast')
plt.xlabel('Day')
plt.ylabel('Number of births')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("Forecasted values:", future_predictions.flatten())

# Plot the loss function
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Testing loss')

plt.title('The loss function')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()