"""
-*-coding: utf-8-*- 
Author: Yann Cherdo 
Creation date: 2020-07-01 14:54:14
"""

import tensorflow as tf 
from libs.utils import weight_variable, bias_variable
import numpy as np
from libs.utils import shuffle_in_unison

class RNN:
    """
    A generic RNN tensorflow implementation allowing one layer and stacked RNN
    using GRU unit and many to many architecture.
    """

    def __init__(self, hidden_units:list, input_dim:int, output_dim:int):
        """[summary]

        Args:
            hidden_units (list): the list of number of RNN unit per stage (for a stacked RNN)
            input_dim (int): the number of dimensions one input data point has
            output_dim (int): the number of dimensions one output data point has
        """        
        
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = weight_variable(shape=[self.hidden_units[-1], output_dim])
        self.biases = bias_variable(shape=[output_dim])
        self.learning_rate = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, None, self.input_dim])
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None, self.output_dim])

        self.pred = self._RNN(self.x)
        self.cost = self._cost()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Creating the op for initializing all variables
        self.init = tf.global_variables_initializer()

        # instantiate a global tf session
        self.sess = tf.Session()
        self.sess.run(self.init)

    def fit(self,
            X_train:np.array,
            Y_train:np.array,
            X_test:np.array,
            Y_test:np.array,
            epochs:int=10,
            mini_batch_size:int=32,
            learning_rate:float=0.01)->tuple:
        """[summary]

        Args:
            X_train (np.array): [description]
            Y_train (np.array): [description]
            X_test (np.array): [description]
            Y_test (np.array): [description]
            epochs (int, optional): [description]. Defaults to 10.
            mini_batch_size (int, optional): [description]. Defaults to 32.
            learning_rate (float, optional): [description]. Defaults to 0.01.

        Returns:
            tuple: [description]
        """        

        train_costs =  []
        test_costs =  []

        # Compute the number of batches in X (adding one if there is a rest)
        n_batch = X_train.shape[0]//mini_batch_size
        if X_train.shape[0]%mini_batch_size > 0:
            n_batch += 1
           
        # print('\n----------Training---------')
        for i in range(epochs):

            epoch_train_cost = []
            epoch_test_cost = []
            for j in range(n_batch):
                # train

                # x_batch, y_batch = random_sample_batch(X_train, Y_train, mini_batch_size)
                x_batch, y_batch = shuffle_in_unison(X_train, Y_train)
                x_batch = x_batch[:mini_batch_size,:,:]
                y_batch = y_batch[:mini_batch_size,:]
                _, train_cost = self.sess.run([self.train_op, self.cost], feed_dict={self.x: x_batch,
                                                                                     self.y: y_batch,
                                                                                     self.learning_rate: learning_rate
                                                                                     })
                epoch_train_cost.append(train_cost)

                # test

                test_cost = self.sess.run([self.cost], feed_dict={self.x: X_test,
                                                                    self.y: Y_test
                                                                    })

                epoch_test_cost.append(test_cost)

            if (i + 1)%1 == 0:
                print('Batch epoch ', i + 1, '/', epochs, ':')
                print('train_cost={0:.4f}, test_cost={1:.4f}'.format(np.mean(epoch_train_cost), np.mean(epoch_test_cost)))
            
            train_costs.extend(epoch_train_cost)
            test_costs.extend(epoch_test_cost)

        return train_costs, test_costs

    def predict(self, X:np.array)->np.array:

        return self.sess.run(self.pred, feed_dict={self.x: X})

    def _cost(self):

        cost = tf.reduce_mean(tf.square(self.pred - self.y))

        return cost

    def _RNN(self, X: np.array):

        cells = [tf.nn.rnn_cell.GRUCell(hu) for hu in self.hidden_units]
        stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, inputs=X, dtype=tf.float32)

        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * tf.shape(outputs)[1] + (tf.shape(X)[1] - 1)
        # Retrieving each sequence last RNN output:
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_units[-1]]), index)

        outputs = tf.matmul(outputs, self.weights) + self.biases # pass from num_hidden to output dimension

        return outputs
