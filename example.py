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
        return self.function(X)

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
        return self.true_fnc(X, Y)

    def F(self, X):
        return self.fake_fnc(X)

    def params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

