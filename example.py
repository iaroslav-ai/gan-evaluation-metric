import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.tensor.shared_randomstreams import RandomStreams

class GNet():
    def __init__(self, input_sz, random_sz, layer_sz, output_sz):
        self.input = T.matrix()
        self.input_sz = input_sz
        self.output_sz = output_sz

        rnd = RandomStreams().normal((self.input.shape[0], random_sz))

        rnd_l = lasagne.layers.InputLayer(shape=(None, random_sz),
                                            input_var=rnd)

        ipt_l = lasagne.layers.InputLayer(shape=(None, input_sz),
                                          input_var=self.input)

        self.network = lasagne.layers.ConcatLayer([rnd_l, ipt_l], axis=1)

        self.network = lasagne.layers.DenseLayer(
                    self.network, layer_sz)

        self.network = lasagne.layers.DenseLayer(
                    self.network, output_sz, nonlinearity=lasagne.nonlinearities.linear)

        self.output = lasagne.layers.get_output(self.network)

        self.function = theano.function(inputs=[self.input], outputs=self.output)

    def __call__(self, X):
        return self.function(X.astype('float32'))

    def params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

class DNet():
    def __init__(self, g, layer_sz):
        self.input = g.input
        self.output = g.output

        input_layer = lasagne.layers.InputLayer(shape=(None, g.input_sz),
                                          input_var=self.input)
        output_layer = lasagne.layers.InputLayer(shape=(None, g.output_sz),
                                          input_var=self.output)

        self.network = lasagne.layers.ConcatLayer([input_layer, output_layer])

        self.network = lasagne.layers.DenseLayer(
                    self.network, layer_sz)

        self.network = lasagne.layers.DenseLayer(
                    self.network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)

        self.fake = lasagne.layers.get_output(self.network)

        self.fake_fnc = theano.function(inputs=[self.input], outputs=self.fake)
        self.true_fnc = theano.function(inputs=[self.input, self.output], outputs=self.fake)

    def T(self, X, Y):
        return self.true_fnc(X.astype('float32'), Y.astype('float32'))

    def F(self, X):
        return self.fake_fnc(X.astype('float32'))

    def params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

import os

def get_data():
    X = np.load(os.path.join('mnist', 'labels.npy')).astype('float32') # gan conditioned on labels
    Y = np.load(os.path.join('mnist', 'images.npy')).astype('float32').astype('float32') # gan conditioned on labels

    Y = np.reshape(Y, (len(Y), -1)) # flatten image
    # onehot the label
    Z = np.zeros( (len(X), len(np.unique(X[:]))) )

    for i in range(len(X)):
        Z[i, X[i][0]] = 1.0

    X = Z.astype('float32')

    X, Xv, Xt = X[:10000], X[10000:15000], X[15000:]
    Y, Yv, Yt = Y[:10000], Y[10000:15000], Y[15000:]

    return X, Xv, Xt, Y, Yv, Yt

X, Xv, Xt, Y, Yv, Yt = get_data()

inputs_size = X.shape[1]
randomness_amount = 32
output_size = Y.shape[1]
dataset_size = 128
layer_size = 128
layer_count = 1
max_iterations = 1024 # max iter for gan training

g = GNet(input_sz=inputs_size, random_sz=randomness_amount, layer_sz=layer_size, output_sz=output_size)
d = DNet(g, layer_sz=layer_size)

import gan_fitter as gf
from matplotlib import pyplot as plt

plt.ion()

def callback(i, g, d):
    if not (i % 10 == 0):
        return

    print i

    if (i % 10 == 0):
        # generate example
        x = np.zeros((1, 10))
        x[0][0] = 1.0
        yp = g(x)
        yp = np.reshape(yp, (16,16))

        plt.imshow(yp)
        plt.show()
        plt.pause(0.01)

gf.fit(g, d, X, Y, max_iterations, callback)

