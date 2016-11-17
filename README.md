# Tests of evaluation metric for GANs

For example evaluation, run example.py. It requires numpy, theano and lasagne to be installed on your system.

The code in the repository works as follows:

1. An example dataset is generated, which contains paris of input / output, where numbers from 0 to 9 are inputs and number image from MNIST dataset as outputs.

2. Dataset is split into training, validation and testing parts.

3. Training part of dataset is used to train GAN using [standard](https://arxiv.org/abs/1606.03498) approach.

4. Using validation dataset, an architecture selection is performed for a classifier which is trained to distinguish between real and generated data points. Architecture selection is performed in grid search manner. For training, equal amount of real and generated data points are taken.

5. Let accuracy of classifier found in previous step be p. Then as performance of GAN the value of 2 - 2p is taken. Such value is 1.0 for GAN which perfectly fools the discriminative classifier, and 0.0 is the worst value. 
