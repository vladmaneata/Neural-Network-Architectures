import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, MaxPooling2D, Flatten

# load data: 60000 train images and 10000 test images
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# plot random chosen number images
for i in range(9):
    ax = plt.subplot(330 + 1 + i)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    rand_img = random.randrange(0, len(train_img))
    plt.title('Label: %i' % train_labels[rand_img], color='b', fontsize=9, fontweight='bold')
    plt.imshow(train_img[rand_img], cmap=plt.get_cmap('gray'))

plt.show()

# define training and testing data
x_train = train_img.reshape(train_img.shape[0], 28, 28, 1)
x_test = test_img.reshape(test_img.shape[0], 28, 28, 1)
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# configure the ConvNet
layers = [
    Conv2D(filters=20, kernel_size=5,  activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(momentum=0.9),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    Flatten(),
    Dense(10,activation='softmax')
]
model = Sequential(layers)
model.summary()

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
history = model.fit(x_train, train_labels, epochs=2, validation_data=(x_test, test_labels))
test_loss, test_acc = model.evaluate(x_test, test_labels, verbose=2)

# plot accuracy vs. epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Training accuracy', 'Testing accuracy'], loc='best')
plt.title('Testing accuracy: %f' %test_acc, fontsize=10, fontweight='bold')
plt.show()

# plot images and label predictions
for i in range(9):
    ax = plt.subplot(330 + 1 + i)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    rand_img = random.randrange(0, len(x_test))
    plt.imshow(x_test[rand_img].reshape(28,28), cmap=plt.get_cmap('gray'))
    y_pred = model.predict(x_test[rand_img].reshape(1, 28, 28, 1))
    predicted_label = np.argmax(y_pred)
    plt.title('Predicted Label: %i' %predicted_label+ '\n Real Value: %i' %test_labels[rand_img], \
              fontsize=9, color='m',fontweight='bold')
plt.show()