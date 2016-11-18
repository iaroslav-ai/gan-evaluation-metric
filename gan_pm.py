import numpy as np
import theano
import theano.tensor as T
import lasagne


class D_OBJ_Optimizer():
    def __init__(self, D):
        self.target = T.vector()

        loss = T.mean((D.output_is_real - self.target)**2)

        updates = lasagne.updates.adam(
            loss, D.params(), learning_rate=0.0001, beta1=0.9)

        self.step = theano.function(inputs=[D.x_input, D.x_output, self.target], outputs=loss, updates=updates)
        self.evaluate = theano.function(inputs=[D.x_input, D.x_output, self.target], outputs=loss)

def split_data(X):
    v = int(len(X) * 0.5)
    t = int(len(X) * 0.75)

    X, Xv, Xt = X[:v], X[v:t], X[t:]

    return X, Xv, Xt

def acc_to_measure(Rt, accuracy):
    # estimate probability of least likely class
    p = min(np.mean(Rt == 1.0), np.mean(Rt == 0.0))
    perf = min(1.0, (1.0 - accuracy) / p)

    return float(perf)

def fitevaluate(Generated, Real, DiscriminatorClass, DiscFixedParams, VariableConfig):
    D = DiscriminatorClass(DiscFixedParams, VariableConfig)

    optimizer = D_OBJ_Optimizer(D)

    Gx, Gy = Generated
    Rx, Ry = Real

    # concatenate everything into one big dataset
    X = np.concatenate([Gx, Rx], axis=0)
    Y = np.concatenate([Gy, Ry], axis=0)

    # indicator variable for whether the instance is real
    R = np.zeros(len(Gx) + len(Rx))
    R[len(Gx):] = 1.0
    R = R.astype('float32')

    # mix everything
    I = np.random.permutation(len(R))
    X = X[I]
    Y = Y[I]
    R = R[I]


    # split the data
    X, Xv, Xt = split_data(X)
    Y, Yv, Yt = split_data(Y)
    R, Rv, Rt = split_data(R)


    # iterate on training data, until validation loss does not improves
    max_patience = 128
    current_patience = max_patience

    batch_size = 256
    best_loss = None
    test_accuracy = None
    total_count = 0 # count of points evaluated

    while current_patience > 0:
        B = np.random.choice(len(X), batch_size)
        optimizer.step(X[B], Y[B], R[B])

        total_count += batch_size

        if total_count > len(X): # epoch is done
            total_count = 0

            loss = optimizer.evaluate(Xv, Yv, Rv)

            if best_loss is None or loss < best_loss:
                best_loss = loss
                # evaluate classifier on test set here
                Rp = D(Xt, Yt)
                Rp = np.round(Rp)
                test_accuracy = np.mean(Rp == Rt)
                current_patience = max_patience
            else:
                current_patience -= 1

            #print 'loss', loss, 'gan performance', acc_to_measure(Rt, test_accuracy)

    # return as quality of gan the value of 2 - 2*test_accuracy
    return acc_to_measure(Rt, test_accuracy)





def score(Generated, Real, DiscriminatorClass, DiscFixedParams, DiscVariableConfig, callback):
    pass
