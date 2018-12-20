import tensorflow as tf
import numpy as np

def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):

    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters],tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias',[out_filters],tf.float32, initializer = tf.zeros_initializer())
        
        return tf.nn.conv2d(x, kernel, [1,strides,strides,1], padding = padding) + bias


    
def Batch_norm(name, x, dim, phase, BN_decay = 0.999, BN_epsilon = 1e-3):
    
    beta = tf.get_variable(name = name + "beta", shape = dim, dtype = tf.float32,
                           initializer = tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name + "gamma", dim, tf.float32, 
                            initializer = tf.constant_initializer(1.0, tf.float32))
    mu = tf.get_variable(name + "mu", dim, tf.float32, 
                         initializer = tf.constant_initializer(0.0, tf.float32), trainable = False)
    sigma = tf.get_variable(name + "sigma", dim, tf.float32, 
                            initializer = tf.constant_initializer(1.0, tf.float32), trainable = False)
    
    if phase is True:
        mean, variance = tf.nn.moments(x, axes = [0, 1, 2])
        train_mean = tf.assign(mu, mu * BN_decay + mean * (1 - BN_decay))
        train_var = tf.assign(sigma, sigma * BN_decay + variance * (1 - BN_decay))
        
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_epsilon)
    else:
        bn_layer = tf.nn.batch_normalization(x, mu, sigma, beta, gamma, BN_epsilon)
        return bn_layer
    
def FC_layer(name, x, input_dim, output_dim):
    with tf.variable_scope(name):
        weight = tf.get_variable(name = 'weight', shape = [input_dim, output_dim], 
                        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
        bias = tf.get_variable(name = 'bias', shape = [output_dim], 
                              initializer = tf.zeros_initializer(), dtype = tf.float32)
        
        return tf.matmul(x, weight) + bias
    
def ResBlock(name, x, filter_size, filters, phase):
    
    _res = x
    
    x = Conv(name = 'Conv1_' + name, x = x, filter_size = filter_size, in_filters = filters, 
             out_filters = filters, strides = 1, padding = 'SAME')
    x = Batch_norm('bn1_' + name, x, filters, phase)
    x = tf.nn.relu(x)
    
    x = Conv(name = 'Conv2_' + name, x = x, filter_size = filter_size, in_filters = filters, 
             out_filters = filters, strides = 1, padding = 'SAME')
    x = Batch_norm('bn2_' + name, x, filters, phase)
    x = x + _res
    
    x = tf.nn.relu(x)
    
    return x

def Batch_norm_train(x, beta, gamma, mu, sigma, BN_decay, BN_epsilon):
    
    mean, variance = tf.nn.moments(x, axes = [0, 1, 2])
    train_mean = tf.assign(mu, mu * BN_decay + mean * (1 - BN_decay))
    train_var = tf.assign(sigma, sigma * BN_decay + variance * (1 - BN_decay))

    with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_epsilon)

def Batch_norm_test(x, beta, gamma, mu, sigma, BN_epsilon):
    bn_layer = tf.nn.batch_normalization(x, mu, sigma, beta, gamma, BN_epsilon)
    return bn_layer
    
def Batch_norm2(name, x, dim, phase, BN_decay = 0.999, BN_epsilon = 1e-3):
    
    beta = tf.get_variable(name = name + "beta", shape = dim, dtype = tf.float32,
                           initializer = tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(name + "gamma", dim, tf.float32, 
                            initializer = tf.constant_initializer(1.0, tf.float32))
    mu = tf.get_variable(name + "mu", dim, tf.float32, 
                         initializer = tf.constant_initializer(0.0, tf.float32), trainable = False)
    sigma = tf.get_variable(name + "sigma", dim, tf.float32, 
                            initializer = tf.constant_initializer(1.0, tf.float32), trainable = False)
    
    return tf.cond(phase, true_fn = lambda : Batch_norm_train(x, beta, gamma, mu, sigma, BN_decay, BN_epsilon),
                   false_fn = lambda : Batch_norm_test(x, beta, gamma, mu, sigma, BN_epsilon))


def ResBlock2(name, x, filter_size, filters, phase):
    
    _res = x
    
    x = Conv(name = 'Conv1_' + name, x = x, filter_size = filter_size, in_filters = filters, 
             out_filters = filters, strides = 1, padding = 'SAME')
    x = Batch_norm2('bn1_' + name, x, filters, phase)
    x = tf.nn.relu(x)
    
    x = Conv(name = 'Conv2_' + name, x = x, filter_size = filter_size, in_filters = filters, 
             out_filters = filters, strides = 1, padding = 'SAME')
    x = Batch_norm2('bn2_' + name, x, filters, phase)
    x = x + _res
    
    x = tf.nn.relu(x)
    
    return x

