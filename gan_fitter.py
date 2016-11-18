import numpy as np
import theano
import theano.tensor as T
import lasagne

class G_OBJ_Optimizer():
    def __init__(self, G, D):
        loss = T.mean((D.output_is_real - 1.0)**2)

        updates = lasagne.updates.nesterov_momentum(
            loss, D.params(), learning_rate=0.01, momentum=0.3)

        self.fnc = theano.function(inputs=[G.input], outputs=loss, updates=updates)

    def __call__(self, X):
        return self.fnc(X)

class D_OBJ_Optimizer():
    def __init__(self, G, D):
        self.target = T.scalar()

        loss = T.mean((D.output_is_real - self.target)**2)

        updates = lasagne.updates.nesterov_momentum(
            loss, D.params(), learning_rate=0.01, momentum=0.3)

        self.fake_fnc = theano.function(inputs=[self.target, G.input], outputs=loss, updates=updates)
        self.true_fnc = theano.function(inputs=[self.target, G.input, G.output], outputs=loss, updates=updates)

    def __call__(self, X, Y):
        l1 = self.fake_fnc(0.0, X)
        l2 = self.true_fnc(1.0, X, Y)
        return l1 + l2

def fit(g, d, x, y, max_iter, callback):

    opt_G = G_OBJ_Optimizer(g, d)
    opt_D = D_OBJ_Optimizer(g, d)

    for i in range(max_iter):
        opt_G(x)
        opt_D(x, y)
        callback(i, g, d)
