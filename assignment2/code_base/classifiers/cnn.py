from builtins import object
import numpy as np

from code_base.layers import *
from code_base.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, dropout=0, seed=123, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.use_dropout = dropout > 0
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim

    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    stride_conv = 1
    pad = filter_size - 1
    h_conv = int((H + pad - filter_size) / stride_conv) + 1
    w_conv = int((W + pad - filter_size) / stride_conv) + 1
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = int((h_conv - height_pool) / stride_pool) + 1
    Wp = int((w_conv - width_pool) / stride_pool) + 1

    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * Hp * Wp, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
    
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Set train/test mode for dropout param since it
    # behaves differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################

    dropout_param1 = {}
    dropout_param1['p'] = 0.7
    dropout_param1['mode'] = mode
    dropout_param2 = {}
    dropout_param2['p'] = 0.5
    dropout_param2['mode'] = mode

    # conv_layer, cache_conv_layer = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    conv_relu_out, conv_relu_cache = conv_relu_forward(X, W1, b1, conv_param)
    dropout1_out, dropout1_cache = dropout_forward(conv_relu_out, dropout_param1)
    mp_out, mp_cache = max_pool_forward(dropout1_out, pool_param)
    # mp_out, mp_cache = max_pool_forward(conv_relu_out, pool_param)

    N, F, Hp, Wp = mp_out.shape
    x = mp_out.reshape((N, F * Hp * Wp))
    fcn_layer, cache_fcn = affine_forward(x, W2, b2)
    dropout2_out, dropout2_cache = dropout_forward(fcn_layer, dropout_param2)
    relu_layer, cache_relu = relu_forward(dropout2_out)
    # relu_layer, cache_relu = relu_forward(fcn_layer)
    cache_hidden_layer = (cache_fcn, cache_relu)
    N, Hh = relu_layer.shape

    scores, cache_scores = affine_forward(relu_layer, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W2 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W3 ** 2)
    loss = data_loss + reg_loss

    # Backpropagation
    grads = {}
    # Backprop into output layer
    dx3, dW3, db3 = affine_backward(dscores, cache_scores)
    dW3 += self.reg * W3

    # Backprop into first layer
    fc_cache, relu_cache = cache_hidden_layer
    da = relu_backward(dx3, relu_cache)
    dq = dropout_backward(da, dropout2_cache)
    dx2, dW2, db2 = affine_backward(dq, fc_cache)
    # dx2, dW2, db2 = affine_backward(da, fc_cache)

    dW2 += self.reg * W2

    # Backprop into the conv layer
    dx2 = dx2.reshape(N, F, Hp, Wp)
    # dx, dW1, db1 = conv_relu_pool_backward(dx2, cache_conv_layer)
    dq2 = max_pool_backward(dx2, mp_cache)
    dq3 = dropout_backward(dq2, dropout1_cache)
    dx, dW1, db1 = conv_relu_backward(dq3, conv_relu_cache)
    # dx, dW1, db1 = conv_relu_backward(dq2, conv_relu_cache)

    dW1 += self.reg * W1

    grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3})

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
