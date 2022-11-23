import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import NeuralNetwork as Net


def image_to_input(path):
    img = Image.open(path)
    img_array = np.array(img)
    gray = 255 - np.mean(img_array[..., :3], -1)
    return np.asfarray(gray).reshape(1, 28*28)


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
    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    # plt.imshow(image_array, cmap='Greys', interpolation='None')
    # plt.show()


def show_image(image):
    image_array = np.asfarray(image).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


def train_net(net, data):
    # Training process
    for record in data:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)
    return net


def calculate_score(net, test_data, show=False):
    scorecard = []
    for record in test_data:
        all_values = record.split(',')
        correct = int(all_values[0])
        inputs = np.asfarray(all_values[1:])
        outputs = net.query(inputs, as_array=True)
        label = np.argmax(outputs)

        if label == correct:
            scorecard.append(1)
        else:
            scorecard.append(0)

        if show:
            print('Correct: ', correct, 'Net: ', label)

    scorecard_array = np.asarray(scorecard)
    efficiency = scorecard_array.sum()/scorecard_array.size
    return scorecard, efficiency


def test_my_data(net):
    files_list = glob.glob('my_images/*.png')
    for file in files_list:
        image_data = image_to_input(file)
        show_image(image_data)
        print('Answer is', net.query(image_data))

train = False
check = True

# Neural network parameters
input_nodes = 28*28
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# Weight_random_function=lambda x,y: np.random.normal(0.0, pow(y,-0.5), (x,y)))
net = Net.NeuralNetwork(input_nodes,
                        hidden_nodes,
                        output_nodes,
                        learning_rate,)
net.load_weights()

if train:
    # Load train data and train net
    net.clear_weights()
    data_train = load_data('mnist_dataset/mnist_train.csv')
    epochs = 5
    for i in range(epochs):
        print('Epoch:', i+1, 'training start')
        train_net(net, data_train)
        net.save_weights()
        print('Epoch:', i+1, 'training complete')

if check:
    # Load test data and calculate efficiency
    data_test = load_data('mnist_dataset/mnist_test.csv')
    score, efficiency = calculate_score(net, data_test)
    print(score, efficiency)

test_my_data(net)