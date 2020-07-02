"""
-*-coding: utf-8-*- 
Author: Yann Cherdo 
Creation date: 2020-06-30 17:56:47
"""

import tensorflow as tf
import numpy as np

def shuffle_in_unison(a:np.array, b:np.array)->tuple:
    """
    shuffle two arrays on same indexes

    Args:
        a (np.array): [description]
        b (np.array): [description]

    Returns:
        tuple: [description]
    """    
    n_elem = a.shape[0]
    indeces = np.random.permutation(n_elem)
    return a[indeces], b[indeces]

def normalize(l)->list:
    """
    Normalise all values of an array in [0, 1]

    Args:
        l ([type]): numerical array-like 

    Returns:
        list: [description]
    """
    v_min = min(l)
    v_max = max(l)
    r = v_max - v_min

    return [(v - v_min)/r if r!=0 else 0 for v in l]

def weight_variable(shape):
    initer = tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32) # tf.truncated_normal_initializer(stddev=0.01)
    return tf.Variable(initer,
        dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.Variable(initial,
        dtype=tf.float32)
