from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Add some random noise to data
noise_level = 0.3;
x_train_noisy = x_train + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_level * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)  # limit the values to (0, 1)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# build the model
encoding_dim = 15
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# Train the model
autoencoder.fit(x_train_noisy, x_train,
                epochs=15,
                batch_size=256,
                validation_data=(x_test_noisy, x_test))

# Do and plot some predictions
decoded_img = autoencoder.predict(x_test_noisy)

plt.figure(figsize=(6, 3))
for i in range(4):
    # Display original
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(x_test_noisy[i+11].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, 4, i + 1 + 4)
    plt.imshow(decoded_img[i+11].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Evaluate the model
evaluation = autoencoder.evaluate(x_test_noisy, x_test, batch_size=128, verbose=1)
print('\n Summary: Loss over the test dataset: %.3f' % evaluation)