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

class G_OBJ_Optimizer():
    def __init__(self, G, D):
        loss = D.fake
        loss = (loss.mean()-1)**2

        updates = lasagne.updates.nesterov_momentum(
            loss, D.params(), learning_rate=0.001, momentum=0.9)

        self.fnc = theano.function(inputs=[G.input], outputs=loss, updates=updates)

    def __call__(self, X):
        return self.fnc(X)

class D_OBJ_Optimizer():
    def __init__(self, G, D):
        self.target = T.scalar()

        loss = D.fake
        loss = (loss.mean()-self.target)**2

        updates = lasagne.updates.nesterov_momentum(
            loss, D.params(), learning_rate=0.001, momentum=0.9)

        self.fake_fnc = theano.function(inputs=[self.target, G.input], outputs=loss, updates=updates)
        self.true_fnc = theano.function(inputs=[self.target, G.input, G.output], outputs=loss, updates=updates)

    def __call__(self, X, Y):
        l1 = self.fake_fnc(0.0, X)
        l2 = self.true_fnc(1.0, X, Y)
        return l1 + l2

g = GNet(input_sz=3, random_sz=4, layer_sz=2, output_sz=2)
d = DNet(g, layer_sz=4)

opt_G = G_OBJ_Optimizer(g, d)
opt_D = D_OBJ_Optimizer(g, d)

x = np.zeros((5,3))
y = np.zeros((5,2))

for i in range(100000):
    opt_D(x, y)
    if i % 1000 == 0:
        print opt_D(x, y)
