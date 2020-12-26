# Author: Petra Vidnerova (https://github.com/PetraVidnerova)
# Source: https://github.com/PetraVidnerova/rbf_keras

import random
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Orthogonal, Constant
import numpy as np


class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        print("--------------X is --------------\n",self.X)
        print(self.X.shape[1])
        print(shape[1])
        assert shape[1] == self.X.shape[1]
        print(self.X.shape[0])
        print(shape[0])
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        print(idx)
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
            # self.initializer = Orthogonal()
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("in build out dim ",self.output_dim," input shape ",input_shape[1])
        print("input shape ", input_shape)
        print("weight size should be (",self.output_dim,",", input_shape[1],")")
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[-1]),
                                       initializer=self.initializer,
                                       trainable=True)
        print("CENTERS ", self.centers)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        """
        C = K.expand_dims(self.centers)
        print("C is ",C)
        H = K.transpose(C - K.transpose(x))
        return K.exp(-self.betas * K.sum(H ** 2, axis=1))
        """
        C = self.centers[np.newaxis, :, :]
        X = x[:, np.newaxis, :]

        diffnorm = K.sum((C-X)**2, axis=-1)
        ret = K.exp( - self.betas * diffnorm)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
