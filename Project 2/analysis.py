import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from PIL import Image
import glob

from tensorflow.python.ops.gen_math_ops import mod


def image_to_input(path):
    img = Image.open(path)
    img_array = np.array(img)
    gray = 255 - np.mean(img_array[..., :3], -1)
    return np.asfarray(gray).reshape(1, 28, 28)


def test_my_data(model, show=True):
    files_list = glob.glob('my_images/*.png')
    for file in files_list:
        image_data = image_to_input(file)
        if show:
            show_image(image_data)
        predictions = model(image_data).numpy()
        # print(y_train[i], '-', np.argmax(tf.nn.softmax(predictions).numpy()))
        print('Answer is', np.argmax(tf.nn.softmax(predictions).numpy()))


def load_data(path):
    # Load train data from file
    data_file = open(path, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def get_input(data, id, show_im=False):
    all_values = data[id].split(',')
    correct = int(all_values[0])
    inputs = np.asfarray(all_values[1:])
    #
    if show_im:
        show_mnist_image(data, id)
    #
    return inputs, correct


def show_mnist_image(data_list, id):
    all_values = data_list[id].split(',')
    image_array = all_values[1:]
    show_image(image_array)


def show_image(image):
    image_array = np.asfarray(image).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


def get_input(data, id, show_im=False):
    all_values = data[id].split(',')
    correct = int(all_values[0])
    inputs = np.asfarray(all_values[1:])
    #
    if show_im:
        show_mnist_image(data, id)
    #
    return inputs, correct


# Prepare data
print("Training and test data preparation started")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Training and test data preparation finished")
# Create model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(256, activation='sigmoid'),
    # layers.Dropout(0.2),
    layers.Dense(10),
])
epochs_array = range(1, 100)
result = []
for epochs in epochs_array:
    # Train
    print("Model training started")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss=loss_fn)
    model.fit(x_train, y_train, epochs=1, )
    model.save('saved_model')
    print("Model training finished")

    print("Model evaluation started")
    res = model.evaluate(x_test,  y_test, verbose=2)
    print("Model evaluation finished, res =", res[1])
    result.append(res[1])

x = np.array(epochs_array)
y = np.array(result)
plt.scatter(x, y)
plt.show()

# for i in range(10):
#     predictions = model((x_train[i, :]).reshape(1, 28, 28)).numpy()
#     print(y_train[i], '-', np.argmax(tf.nn.softmax(predictions).numpy()))

# test_my_data(model, False)
