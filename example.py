import numpy as np
import theano
import theano.tensor as T
import lasagne

# architecture of discriminator network used for evaluation
class CNet():
    def __init__(self, fixed_params, variable_params):
        self.input_sz = fixed_params['input_sz']
        self.output_sz = fixed_params['output_sz']

        self.layer_sz = round( 2.0 ** (variable_params['layer_sz']) )

        self.x_input = T.matrix()
        self.x_output = T.matrix()

        ip_layer = lasagne.layers.InputLayer(shape=(None, self.input_sz),
                                            input_var=self.x_input)

        op_layer = lasagne.layers.InputLayer(shape=(None, self.output_sz),
                                          input_var=self.x_output)

        self.network = lasagne.layers.ConcatLayer([ip_layer, op_layer], axis=1)

        self.network = lasagne.layers.DenseLayer(
                    self.network, self.layer_sz)

        self.network = lasagne.layers.DenseLayer(
                    self.network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)

        self.output_is_real = lasagne.layers.get_output(self.network)[:, 0]

        self.function = theano.function(inputs=[self.x_input, self.x_output], outputs=self.output_is_real)

    def __call__(self, X, Y):
        return self.function(X.astype('float32'), Y.astype('float32'))

    def params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)



import os

def get_data():
    X = np.load(os.path.join('mnist', 'labels.npy')).astype('float32') # gan conditioned on labels
    Y = np.load(os.path.join('mnist', 'images.npy')).astype('float32').astype('float32') # gan conditioned on labels

    Y = np.reshape(Y, (len(Y), -1)).astype('float32') # flatten image
    # onehot the label
    Z = np.zeros( (len(X), len(np.unique(X[:]))) )

    for i in range(len(X)):
        Z[i, int(X[i][0])] = 1.0

    X = Z.astype('float32')

    return X, Y

X, Y = get_data()

import gan_pm

inputs_size = X.shape[1]
outputs_size = Y.shape[1]

fixed_params = {'input_sz': inputs_size, 'output_sz': outputs_size}
variable_config = {'layer_sz':{'type': 'real', 'bounds':[2.0, 10.0]}}

# known parameters that yield good performance of discriminator. should be found by grid search
known_good_parameters = {'layer_sz': 8.0}

print "This script usually takes few minutes on GPU to compute."
print ""

perf = gan_pm.fitevaluate((X, Y), (X, Y), CNet, fixed_params, known_good_parameters)
print "Performance with perfect GAN - generated data same as actual data"
print perf

"""
Warning: score above you are likely to get if your GAN overfitted to the training dataset
AND you use training dataset for evaluation of GAN
"""

perf = gan_pm.fitevaluate((X, Y), (X, np.zeros_like(Y)), CNet, fixed_params, known_good_parameters)
print "Performance with very bad GAN - generated data is all zeros"
print perf

print "Simulating GAN outputs by a set of digit images with label less than some value. This will go from bad to good GAN."

for i in range(10):

    L = len(X) // 2

    Xa, Xb = X[:L], X[L:]
    Ya, Yb = Y[:L], Y[L:]

    I = np.argmax(Xa, axis=1) <= i

    Xa = Xa[I]
    Ya = Ya[I]

    Xb = Xb[I]
    Yb = Yb[I]

    perf = gan_pm.fitevaluate((Xa, Ya), (Xb, Yb), CNet, fixed_params, known_good_parameters)
    print "Performance of gan that generates digits less equal to ", str(i)
    print perf

print "Testing with subsets of digit labels in generated data. This will go from bad to good GAN."