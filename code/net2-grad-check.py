import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# np.random.seed(1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
import csv


def load_data():
    X_train = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]).astype(np.float128)
    T_train = np.array([[0, 0, 0, 1]])

    return (X_train, T_train)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def relu(z):
    return np.maximum(z, 0)


def relu_deriv(y):
    return 1. * (y > 0)


# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""

    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []

    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass

    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the
         output error instead of output_grad"""
        pass


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)

    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)


class ReluLayer(Layer):
    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(relu_deriv(Y), output_grad)


class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)

    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]

    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]


# Define the forward propagation step as a method.
def forward_step(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.
    """
    activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  # Get the output of the current layer
        activations.append(Y)  # Store the output for future processing
        X = activations[-1]  # Set the current input as the activations of the previous layer
    return activations  # Return the activations of each layer


# Define the backward propagation step as a method
def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at
            each index i+1 corresponds to the activation of layer i in layers.
            activations[0] contains the input samples.
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers.
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    #  Use reversed to iterate backwards over the list of layers.
    for layer in reversed(layers):
        Y = activations.pop()  # Get the activations of the last layer on the stack
        # Compute the error at the output layer.
        # The output layer error is calculated different then hidden layer error.
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:  # output_grad is not None (layer is not output layer)
            input_grad = layer.get_input_grad(Y, output_grad)
        # Get the input of this layer (activations of the previous layer)
        X = activations[-1]
        # Compute the layer parameter gradients used to update the parameters
        grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)  # Return the parameter gradients


# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in zip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter


if __name__ == '__main__':
    X_train, T_train = load_data()

    # Define a sample model to be trained on the data
    hidden_neurons = 28  # Number of neurons in the first hidden-layer
    # Create the model
    layers = []  # Define a list of layers
    # Add first hidden layer
    layers.append(LinearLayer(X_train.shape[1], hidden_neurons))
    layers.append(ReluLayer())
    # Add second hidden layer
    layers.append(LinearLayer(hidden_neurons, hidden_neurons))
    layers.append(ReluLayer())
    # Add third hidden layer
    layers.append(LinearLayer(hidden_neurons, hidden_neurons))
    layers.append(ReluLayer())
    # Add fourth hidden layer
    layers.append(LinearLayer(hidden_neurons, hidden_neurons))
    layers.append(ReluLayer())
    # Add fifth hidden layer
    layers.append(LinearLayer(hidden_neurons, hidden_neurons))
    layers.append(ReluLayer())
    # Add sixth hidden layer
    layers.append(LinearLayer(hidden_neurons, hidden_neurons))
    layers.append(ReluLayer())
    # Add output layer
    layers.append(LinearLayer(hidden_neurons, T_train.shape[1]))
    layers.append(SoftmaxOutputLayer())

    with open(
            '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_4/c/w-28-6-4.csv') as csvfile:
        r = csv.reader(csvfile, delimiter='\n', quotechar='|')
        i = 0
        j = 0
        end = 0
        for row in r:
            str = row[0].split(',')
            str = str[1:]
            arr = list(map(float, str))
            arr = np.asarray(arr)
            layers[j].W[i] = arr
            i += 1

            for k in range(len(layers) / 2):
                if j == (k * 2) and i == layers[j].W.shape[0]:
                    if k == ((len(layers) / 2) - 1):
                        end = 1
                        break
                    j += 2
                    i = 0
                    break
            if end == 1:
                break

        with open(
                '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_4/c/b-28-6-4.csv') as csvfile:
            r = csv.reader(csvfile, delimiter='\n', quotechar='|')
            i = 0
            j = 0
            for row in r:
                str = row[0].split(',')
                str = str[1:]
                arr = list(map(float, str))
                arr = np.asarray(arr)
                layers[j].b = arr
                j += 2
                if j == len(layers):
                    break

    # Perform gradient checking
    nb_samples_gradientcheck = 1  # Test the gradients on a subset of the data
    X_temp = X_train[0:nb_samples_gradientcheck, :]
    T_temp = T_train[0:nb_samples_gradientcheck, :]
    # Get the parameter gradients with backpropagation
    activations = forward_step(X_temp, layers)
    param_grads = backward_step(activations, T_temp, layers)

    with open('dW-28-6-4.csv', 'wb') as csvfile:
        myWriter = csv.writer(csvfile)
        for j in range((len(layers) / 2)):
            for i in range(layers[j * 2].W.shape[0]):
                myWriter.writerow(param_grads[j * 2][(i * layers[j * 2].W.shape[1]):((i + 1) * layers[j * 2].W.shape[1])])

    with open('db-28-6-4.csv', 'wb') as csvfile:
        myWriter = csv.writer(csvfile)

        for i in range((len(layers) / 2)):
            start_nb = layers[i * 2].W.shape[0] * layers[i * 2].W.shape[1]
            myWriter.writerow(param_grads[i * 2][start_nb:])
