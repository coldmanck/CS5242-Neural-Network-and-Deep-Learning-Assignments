import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# np.random.seed(1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
import csv


def load_data():
    X_train = []
    X_test = []
    T_train = []
    T_test = []

    with open(
            '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_123/x_train.csv') as csvfile:
        r = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in r:
            str = row[0].split(',')
            arr = list(map(int, str))
            X_train.append(arr)
    X_train = np.asarray(X_train)

    with open(
            '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_123/x_test.csv') as csvfile:
        r = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in r:
            str = row[0].split(',')
            arr = list(map(int, str))
            X_test.append(arr)
    X_test = np.asarray(X_test)

    with open(
            '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_123/y_train.csv') as csvfile:
        r = csv.reader(csvfile, delimiter='\n')
        for row in r:
            if row[0] == '0':
                T_train.append([1, 0, 0, 0])
            elif row[0] == '1':
                T_train.append([0, 1, 0, 0])
            elif row[0] == '2':
                T_train.append([0, 0, 1, 0])
            elif row[0] == '3':
                T_train.append([0, 0, 0, 1])
    T_train = np.asarray(T_train)

    with open(
            '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_123/y_test.csv') as csvfile:
        r = csv.reader(csvfile, delimiter='\n')
        for row in r:
            if row[0] == '0':
                T_test.append([1, 0, 0, 0])
            elif row[0] == '1':
                T_test.append([0, 1, 0, 0])
            elif row[0] == '2':
                T_test.append([0, 0, 1, 0])
            elif row[0] == '3':
                T_test.append([0, 0, 0, 1])
    T_test = np.asarray(T_test)

    # Divide the test set into a validation set and final test set.
    X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
        X_test, T_test, test_size=0.4)

    return (X_train, X_test, X_validation, T_train, T_test, T_validation)


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


def gradient_check(layers):
    # Perform gradient checking
    nb_samples_gradientcheck = 10 # Test the gradients on a subset of the data
    X_temp = X_train[0:nb_samples_gradientcheck,:]
    T_temp = T_train[0:nb_samples_gradientcheck,:]
    # Get the parameter gradients with backpropagation
    activations = forward_step(X_temp, layers)
    param_grads = backward_step(activations, T_temp, layers)

    # Set the small change to compute the numerical gradient
    eps = 0.0001
    # Compute the numerical gradients of the parameters in all layers.
    for idx in range(len(layers)):
        layer = layers[idx]
        layer_backprop_grads = param_grads[idx]
        # Compute the numerical gradient for each parameter in the layer
        for p_idx, param in enumerate(layer.get_params_iter()):
            grad_backprop = layer_backprop_grads[p_idx]
            # + eps
            param += eps
            plus_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # - eps
            param -= 2 * eps
            min_cost = layers[-1].get_cost(forward_step(X_temp, layers)[-1], T_temp)
            # reset param value
            param += eps
            # calculate numerical gradient
            grad_num = (plus_cost - min_cost)/(2*eps)
            # Raise error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_backprop):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_backprop)))
    print('No gradient errors found')


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



def plot_costs(minibatch_costs, training_costs, validation_costs, nb_of_iterations, nb_of_batches):
    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations * nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    # Plot the cost over the iterations
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, nb_of_iterations, 0, 2.5))
    plt.grid()
    plt.show()


def plot_accuracys(train_accuracys, validation_accuracys, nb_of_iterations):
    # Plot the minibatch, full training set, and validation costs
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    # Plot the cost over the iterations
    plt.plot(iteration_x_inds, train_accuracys, 'r-', linewidth=2, label='acc. full training set')
    plt.plot(iteration_x_inds, validation_accuracys, 'b-', linewidth=3, label='acc. validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('Increase of accuracy over backprop iteration')
    plt.legend(loc=4)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((0, nb_of_iterations, 0, 1.0))
    plt.grid()
    plt.show()


if __name__ == '__main__':
    X_train, X_test, X_validation, T_train, T_test, T_validation = load_data()

    # Define a sample model to be trained on the data
    hidden_neurons_1 = 100  # Number of neurons in the first hidden-layer
    hidden_neurons_2 = 40  # Number of neurons in the second hidden-layer
    # Create the model
    layers = []  # Define a list of layers
    # Add first hidden layer
    layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1))
    layers.append(ReluLayer())
    # Add second hidden layer
    layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2))
    layers.append(ReluLayer())
    # Add output layer
    layers.append(LinearLayer(hidden_neurons_2, T_train.shape[1]))
    layers.append(SoftmaxOutputLayer())

    gradient_check(layers)

    # Create the minibatches
    batch_size = 25  # Approximately 25 samples per batch
    nb_of_batches = X_train.shape[0] / batch_size  # Number of batches
    # Create batches (X,Y) from the training set
    XT_batches = zip(
        np.array_split(X_train, nb_of_batches, axis=0),  # X samples
        np.array_split(T_train, nb_of_batches, axis=0))  # Y targets

    # Perform backpropagation
    # initalize some lists to store the cost for future analysis
    minibatch_costs = []
    training_costs = []
    validation_costs = []
    train_accuracys = []
    validation_accuracys = []

    max_nb_of_iterations = 100  # Train for a maximum of 300 iterations
    learning_rate = 0.01  # Gradient descent learning rate

    y_true = np.argmax(T_test, axis=1)  # Get the target outputs
    x_train_true = np.argmax(T_train, axis=1)
    x_val_true = np.argmax(T_validation, axis=1)

    # Train for the maximum number of iterations
    for iteration in range(max_nb_of_iterations):
        for X, T in XT_batches:  # For each minibatch sub-iteration
            activations = forward_step(X, layers)  # Get the activations
            minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
            minibatch_costs.append(minibatch_cost)
            param_grads = backward_step(activations, T, layers)  # Get the gradients
            update_params(layers, param_grads, learning_rate)  # Update the parameters

        # Get full training cost for future analysis (plots)
        activations = forward_step(X_train, layers)
        train_cost = layers[-1].get_cost(activations[-1], T_train)
        training_costs.append(train_cost)
        y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
        train_accuracy = metrics.accuracy_score(x_train_true, y_pred)  # Test set accuracy
        train_accuracys.append(train_accuracy)

        # Get full validation cost
        activations = forward_step(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], T_validation)
        validation_costs.append(validation_cost)
        y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
        validation_accuracy = metrics.accuracy_score(x_val_true, y_pred)  # Test set accuracy
        validation_accuracys.append(validation_accuracy)

        print('iter {}: train loss {:.4f} acc {:.4f}, val loss {:.4f} acc {:.4f}'.format(iteration + 1, train_cost, train_accuracy, validation_cost, validation_accuracy))

        if len(validation_costs) > 3:
            # Stop training if the cost on the validation set doesn't decrease
            #  for 3 iterations
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break

    nb_of_iterations = iteration + 1  # The number of iterations that have been executed

    plot_costs(minibatch_costs, training_costs, validation_costs, nb_of_iterations, nb_of_batches)
    plot_accuracys(train_accuracys, validation_accuracys, nb_of_iterations)

    # Get results of test data
    activations = forward_step(X_test, layers)  # Get activation of test samples
    y_pred = np.argmax(activations[-1], axis=1)  # Get the predictions made by the network
    test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
    print('The final accuracy on the test set is {:.4f}'.format(test_accuracy))