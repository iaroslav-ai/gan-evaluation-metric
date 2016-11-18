# Tests of evaluation metric for GANs

For example evaluations using MNIST dataset, run example.py. It requires numpy, theano and lasagne to be installed on your system.

Note: gan as such is not required for example evaluations to work, only its output. Such outputs are simulated
by perturbing the original MNIST dataset. 

Proposed scheme of training / evaluation of gan:

1. An example dataset is generated, which contains paris of input / output for GAN, where numbers from 0 to 9 are inputs and number image from MNIST dataset as outputs.

2. Dataset is split into training, validation and testing parts.

3. Training part of dataset is used to train GAN using [standard](https://arxiv.org/abs/1606.03498) approach.

4. Using validation dataset, an architecture selection is performed for a classifier which is trained to distinguish between real and generated. Architecture selection is performed in grid search manner. For training, equal amount of real and generated data points are taken.

5. Let accuracy on test set of classifier found in previous step be p. Then as performance of GAN the value of (1.0 - p)/l is taken, where l is a frequency of least probable class. Such value is 1.0 for GAN which perfectly fools the discriminative classifier, and 0.0 is the worst value. 
