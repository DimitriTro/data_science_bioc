"""
-*-coding: utf-8-*- 
Author: Yann Cherdo 
Creation date: 2020-06-30 17:56:57
"""

import tensorflow as tf 
from libs.utils import weight_variable, bias_variable
from libs.utils import random_split_indexes, random_sample_batch
import numpy as np
from sklearn.metrics import f1_score


class Abstract_RNN_manyToOne_dynamical:
    """
    A many to one RNN architecture that allows for variable size of sequences in input. The input has to
    be padded and an array seq_len has to inform the real size of each sequence.
    """

    def close_session(self):

        self.sess.close()

        print('\nModel tf session closed.')


    def fit(self,
            X_train,
            Y_train,
            X_test,
            Y_test,
            seq_len_train,
            seq_len_test,
            validation_split=0.1,
            epochs = 10,
            mini_batch_size = 32,
            learning_rate = 0.01):
        """
        param X_train: (array) shape (samples, max_seq_len, input_dim)
        param Y_train: (array) shape (samples, 1, output_dim)
        param X_test: (array) shape (samples, max_seq_len, input_dim)
        param Y_test: (array) shape (samples, 1, output_dim)
        param seq_len_train: (array) the lenght of each sequence in X   
        param seq_len_test: (array) the lenght of each sequence in X    
        param learning_rate: (float) The optimization initial learning rate
        param epochs: (int)
        param mini_batch_size: (int) mini batch size
        param learning_rate: (float)
        return: (array) training costs, (array) testin costs
        """

        self._check_input_shape(X_train)
        self._check_output_shape(Y_train)
        self._check_input_shape(X_test)
        self._check_output_shape(Y_test)

        # train_indexes, valid_indexes = random_split_indexes(X.shape[0], validation_split)

        # x_train, y_train, seq_len_train = X[train_indexes], Y[train_indexes], seq_len[train_indexes] 
        # x_valid, y_valid, seq_len_valid = X[valid_indexes], Y[valid_indexes], seq_len[valid_indexes] 

        train_costs =  []
        test_costs =  []

        # init_hidden_state = np.zeros((mini_batch_size, self.num_hidden_units))
        # init_context_state = np.zeros((mini_batch_size, self.num_hidden_units))

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

                x_batch, y_batch, seq_len_batch = random_sample_batch(X_train, Y_train, seq_len_train, mini_batch_size)
                _, train_cost = self.sess.run([self.train_op, self.cost], feed_dict={self.x: x_batch,
                                                                                     self.y: y_batch,
                                                                                     self.seqLen: seq_len_batch,
                                                                                     self.learning_rate: learning_rate
                                                                                     # self.hidden_state: init_hidden_state,
                                                                                     # self.context_state: init_context_state
                                                                                     })
                epoch_train_cost.append(train_cost)

                # test

                test_cost = self.sess.run([self.cost], feed_dict={self.x: X_test,
                                                                    self.y: Y_test,
                                                                    self.seqLen: seq_len_test
                                                                    })

                epoch_test_cost.append(test_cost)

            if (i + 1)%1 == 0:
                print('Batch epoch ', i + 1, '/', epochs, ':')
                print('train_cost={0:.4f}, test_cost={1:.4f}'.format(np.mean(epoch_train_cost), np.mean(epoch_test_cost)))
            
            train_costs.extend(epoch_train_cost)
            test_costs.extend(epoch_test_cost)

        return train_costs, test_costs


    def cost(self, X, Y, seq_len):
        """
        Obtain cost without training
        param X: (array) shape (samples, max_seq_len, input_dim)
        param Y: (array) shape (samples, 1, output_dim)
        param seq_len: (array) the lenght of each sequence in X    
        return: (array) costs   
        """

        cost = self.sess.run([self.cost], feed_dict={self.x: X,
                                                    self.y: Y,
                                                    self.seqLen: seq_len
                                                    })


    def iterative_predict(self, X, seq_len, iterative_prediction_span=1):
        """
        predict multiple time steps by using last predictions
        /!\ works only if output_dim = input_dim because we reinject the prediction in the input
        param X: (array) shape (batch_size, max_seq_len, input_dim)
        param seq_len: (array) X corresponding sequences lengths
        param iterative_prediction_span: (int) the number of step to predict after each sequences
        return (array) of shape (batch_size, iterative_prediction_span, input_dim)
        """

        self._check_input_shape(X)

        assert self.output_dim == self.input_dim, "To use iterative predictions, the output dimension must equal to the input dimemsion. Got input_dim={0} and output_dim={1}".format(self.input_dim, self.output_dim)

        batch_size = X.shape[0]
        predictions = []
        one_pred = X 

        for i in range(iterative_prediction_span):

            # At first step prediction start without inital state on given sequences X. After first prediction, get next prediction by using last RNN state

            one_pred, last_state = self.sess.run([self.pred, self.last_state], feed_dict={self.x: one_pred, self.seqLen: seq_len})
            
            self.initial_state = self.last_state  # self.initial_state is directly used in method _RNN where self.last_state is updated at each call

            # From now on we predict step t using step t-1 and RNN states at t-1        

            one_pred = one_pred.reshape((batch_size, 1, self.input_dim))

            seq_len = np.ones(shape=(batch_size,)) 

            predictions.append(one_pred)
        
        predictions = np.array(predictions).reshape((batch_size, iterative_prediction_span, self.output_dim))

        self.initial_state = None  # reset initial state

        return predictions


    def predict(self, X, seq_len):
        """
        param X: (array) shape (batch_size, max_seq_len, input_dim)
        param seq_len: (array) X corresponding sequences lengths
        return (array) of shape (batch_size, input_dim)
        """

        self._check_input_shape(X)

        return self.sess.run(self.pred, feed_dict={self.x: X, self.seqLen: seq_len})


    def load_session(self, path):
        """
        param path: (str)
        """

        self.saver.restore(self.sess, path)

        print('Session loaded from: ', path)


    def save_session(self, path):
        """
        param path: (str)
        """

        self.saver.save(self.sess, path)

        print('Session saved at: ', path)


    def use_gpu(self):
        # TODO implement
        # tf.config.experimental.list_physical_devices('GPU')

        print('\nTensorflow has been set to use gpu.')


    def _check_input_shape(self, X):

        assert len(X.shape) == 3 and X.shape[1] == self.seq_max_len and X.shape[2] == self.input_dim, "Input X should be of shape: (-1, {0}, {1}), got: {2}.".format(self.seq_max_len, self.input_dim, X.shape)


    def _check_output_shape(self, Y):

        assert len(Y.shape) == 2 and Y.shape[1] == self.output_dim, "Output Y should be of shape: (-1, {0}), got: {1}.".format(self.output_dim, Y.shape)


class RNN_manyToOne_dynamical_subActivations_subLosses(Abstract_RNN_manyToOne_dynamical):


    def generate_graph(self,
                input_dim,           
                seq_max_len,        
                output_dim,
                sub_dims,
                sub_losses,
                sub_activations,
                hidden_units
                ):
        """
        param input_dim: (int) input dimension
        param seq_max_len: (int) sequence maximum length
        param output_dim: (int) output dimension
        param hidden_units: (list) number of units in each layer ex: [100, 50] --> a stacked LSTM with 100 units on first layer, 50 on second layer
        param sub_dims: (list) list of of sub dimensions in a sample
        param sub_losses: (dict) loss type (str) either 'L2MSE' or 'MSE': (list) corresponding indexes in sub_dims
        param sub_activations: (dict) activation type (str) either 'sigmoid', 'softmax' or 'linear': (list) corresponding indexes in sub_dims
        """

        print('\nGenerating TF graph.')

        tf.reset_default_graph() # Make sure not get in conflict with an existing kernel (TODO might cause problems for instantiating multiple objects...)

        self.input_dim = input_dim
        self.seq_max_len = seq_max_len
        self.output_dim = output_dim
        self.hidden_units = hidden_units

        # Check sub dims, sub losses, sub activations consistency
        valid_losses = {'L2MSE', 'MSE', 'CROSSENTROPY', 'BINARYCROSSENTROPY'}
        valid_activations = {'sigmoid', 'linear', 'softmax'}
        assert np.all([loss in valid_losses for loss in list(sub_losses.keys())]), "Losses should be in {0}, got {1}.".format(valid_losses, list(sub_losses.keys()))
        assert np.all([activation in valid_activations for activation in list(sub_activations.keys())]), "Activations should be in {0}, got {1}.".format(valid_activations, list(sub_activations.keys()))
        assert len(sub_dims) == np.sum([len(indexes) for indexes in sub_losses.values()]), "Each sub dimension corresponding index in sub_dims should be found in a loss in sub_losses, got sub_losses: {0} and sub_dims: {1}".format(sub_losses, sub_dims)
        assert len(sub_dims) == np.sum([len(indexes) for indexes in sub_activations.values()]), "Each sub dimension corresponding index in sub_dims should be found in an activation in sub_activations, got sub_activations: {0} and sub_dims: {1}".format(sub_activations, sub_dims)

        self.sub_dims = sub_dims
        self.sub_losses = sub_losses
        self.sub_activations = sub_activations

        self.weights = weight_variable(shape=[self.hidden_units[-1], output_dim])
        self.biases = bias_variable(shape=[output_dim])
        # self.context_state = tf.placeholder(tf.float32, [None, self.num_hidden_units])
        # self.hidden_state = tf.placeholder(tf.float32, [None, self.num_hidden_units])
        # self.init_state = tf.nn.rnn_cell.LSTMStateTuple(self.context_state, self.hidden_state)

        self.epsilon = tf.constant(value=0.0000000001) # used in cross-entropy loss to avoid float 
        # uncertainty due to sigmoid exp which can cause log to be taken at zero. 10^-10 corresponds to a float 32

        self.learning_rate = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32, [None, None, self.input_dim])
        self.seqLen = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None, self.output_dim])
        self.last_state = None
        self.initial_state = None

        # Network predictions
        # self.pred_train, _ = self._RNN(self.x, self.seqLen)
        self.pred = self._RNN(self.x, self.seqLen)

        # Define the loss function and optimizer
        # self.job_emb_cost = tf.losses.cosine_distance(self.y[:,:job_emb], self.pred[:,:job_emb])
        # self.duration_cost = tf.square(self.y[:, job_emb] - self.pred[:, job_emb])
        # self.cost = tf.linalg.matmul(self.features_weights, tf.concat([self.job_emb_cost, self.duration_cost]), transpose_b=True)
        self.cost = self._cost()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Creating the op for initializing all variables
        self.init = tf.global_variables_initializer()

        # instantiate a global tf session
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.saver = tf.train.Saver()


    def _cost(self):
        """
        linear combination of different adapted costs
        available costs: L2MSE, MSE, crossentropy
        """
        
        costs = []
        pred_splits = tf.split(self.pred, self.sub_dims, 1)
        label_splits = tf.split(self.y, self.sub_dims, 1)

        for index in self.sub_losses['L2MSE']:

            costs.append(tf.reduce_mean(tf.square(tf.math.l2_normalize(pred_splits[index]) - label_splits[index])))

        for index in self.sub_losses['MSE']:

            costs.append(tf.reduce_mean(tf.square(pred_splits[index] - label_splits[index])))

        for index in self.sub_losses['BINARYCROSSENTROPY']:

            # bce = tf.keras.losses.BinaryCrossentropy()
            # costs.append(bce(label_splits[index], pred_splits[index]))
            # costs.append(bce(label_splits[index], pred_splits[index]) + tf.math.abs(tf.norm(pred_splits[index])-1)) # regularized version
            costs.append(tf.reduce_mean(-tf.multiply(label_splits[index], tf.log(pred_splits[index] + self.epsilon)) - tf.multiply(1 - label_splits[index], tf.log(1 - pred_splits[index] + self.epsilon))))

        for index in self.sub_losses['CROSSENTROPY']:

            costs.append(tf.reduce_mean(-tf.multiply(label_splits[index], tf.log(pred_splits[index] + self.epsilon))))

        cost = tf.reduce_mean(costs)

        return cost


    def _RNN(self, X, seq_len):
        """
        param X: (array) inputs of shape [batch_size, max_time, input_dim]
        param seq_len: (array) length of each sequence of shape [batch_size,]
        return (array) of shape (batch_size, output_dim)
        """

        # cells = [tf.nn.rnn_cell.BasicLSTMCell(hu) for hu in self.hidden_units]
        cells = [tf.nn.rnn_cell.GRUCell(hu) for hu in self.hidden_units]
        stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
        outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, inputs=X, sequence_length=seq_len, dtype=tf.float32)

        # RNN outputs is shape (batch_size, max_seq_len, n_hidden), we pass it to shape (-1, n_hidden) and retrieve
        # only samples corresponding to each sequence last RNN output
        # Last indices for each sequence:
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * tf.shape(outputs)[1] + (seq_len - 1)
        # Retrieving each sequence last RNN output:
        outputs = tf.gather(tf.reshape(outputs, [-1, self.hidden_units[-1]]), index)

        outputs = tf.matmul(outputs, self.weights) + self.biases # pass from num_hidden to output dimension

        out_splits = tf.split(outputs, self.sub_dims, 1)

        outs = [None]*len(self.sub_dims) # Order of sub vectors in original vector must be kept!
        # thus we alocate each transformed sub vector to its corresponding order (index) in the original vector
        for index in self.sub_activations['sigmoid']:
            outs[index] = tf.math.sigmoid(out_splits[index])

        for index in self.sub_activations['softmax']:
            outs[index] = tf.nn.softmax(out_splits[index])

        for index in self.sub_activations['linear']:
            outs[index] = out_splits[index]

        outputs = tf.concat(outs, 1)

        return outputs


    def generate_nhots_thresholds(self, Y, preds, res=100):
        """
        Generate one threshold per sub vector using sigmoid crossentropy (nhots) in same order than self.sub_dims
        We obtain a threshold by maximizing F1 score between Y and (preds>threshold)*1
        param Y: (array)
        param preds: (array)
        param res: (int) the number of thresholds to consider in the range [0, 1]
        return (list of obj with predict method)
        """

        print('\nGenerating n hots thresholds.')

        thresholds = []
        last_dim = 0

        class Threshold:
            def __init__(self, threshold):
                self.threshold = threshold
            def predict(self, Y):
                return (Y > self.threshold)*1

        class Max_softmax:
            def predict(self, Y):
                return np.eye(Y.shape[-1])[np.argmax(Y, 1)]

        for index, sub_dim in enumerate(self.sub_dims):

            #if index in self.sub_activations['softmax']: # one hot
            #    thresholds.append(Max_softmax())

            if index in self.sub_losses['CROSSENTROPY'] or index in self.sub_losses['BINARYCROSSENTROPY']: # n hots
                Y_sub = Y[:, last_dim: last_dim + sub_dim].reshape(-1)
                preds_sub = preds[:, last_dim: last_dim + sub_dim].reshape(-1)

                # Search threshold that maximize F1 score
                best_f1 = 0
                best_threshold = 1
                for threshold in [t*(1/res) for t in range(res)]:
                    onehot_preds_sub = (preds_sub > threshold)*1
                    f1 = f1_score(Y_sub, onehot_preds_sub)
                    if f1 > best_f1:
                        best_threshold = threshold
                        best_f1 = f1

                thresholds.append(Threshold(best_threshold))


            else:
                thresholds.append(None)

            last_dim += sub_dim

        return thresholds
