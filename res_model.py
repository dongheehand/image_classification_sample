import tensorflow as tf
from ops import *

class model():
    
    def __init__(self, args):
        
        self.width = args.width
        self.height = args.height
        self.num_label = args.num_label
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        
        self.x = tf.placeholder(name = 'input', shape = [None, self.height, self.width, 3], dtype = tf.float32)
        self.y = tf.placeholder(name = 'label', shape = [None, self.num_label], dtype = tf.float32)
        self.phase = tf.placeholder(name = 'phase', shape = None, dtype = tf.bool)
        self.global_step = tf.placeholder(name = 'learning_step', shape = None, dtype = tf.int32)
        
    def build_model(self):
        
        x = (self.x / 255.0) - 0.5
        
        x = Conv('conv_first', x, 7, 3, 32, 2, "SAME")
        
        for i in range(2):
            x = ResBlock2('ResBlock_0_%02d'%i, x, 3, 32, self.phase)
        
        x = Conv('conv_2nd', x, 3, 32, 64, 2, "SAME")
        
        for i in range(3):
            x = ResBlock2('ResBlock_1_%02d'%i, x, 3, 64, self.phase)
            
        x = Conv('conv_3rd', x, 3, 64, 128, 2, "SAME")
        
        for i in range(2):
            x = ResBlock2('ResBlock_2_%02d'%i, x, 3, 128, self.phase)
        
        self.feature_map = x
        
        x = tf.reduce_mean(x, axis = [1,2])
        
        x = FC_layer('classify01', x, 128, 256)
        x = tf.nn.relu(x)
        self.label = FC_layer('classify02', x, 256, 10)
        
        self.acc = self.accuracy(self.y, self.label)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.label, labels = self.y))
        lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate ,staircase = True)
        self.train = tf.train.AdamOptimizer(learning_rate = lr).minimize(self.loss)

        logging_loss = tf.summary.scalar(name = 'train_loss', tensor = self.loss)
        logging_acc = tf.summary.scalar(name = 'train_acc', tensor = self.acc[0])
        
    def model_parameter(self):
        total = 0
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for var in variables:
            shape = var.shape
            c = 1
            for k in shape:
                c = c * k
            total += c
        
        self.param_num = total
        
        return total
    
    def accuracy(self, logits, labels):
        
        logit_index = tf.argmax(logits, axis = 1)
        labels_index = tf.argmax(labels, axis = 1)
        is_true = tf.cast(tf.equal(logit_index, labels_index), tf.float32)
        acc = tf.reduce_mean(is_true)
        
        return acc, is_true
    
    def grad_cam(self):        
        max_index = tf.argmax(self.label, axis = 1)
        max_one_hot = tf.one_hot(max_index, depth = self.num_label)
        score = tf.reduce_sum(tf.multiply(max_one_hot, self.label), axis = 1)
        
        alpha = tf.reduce_mean(tf.gradients([score], self.feature_map)[0], axis = [1,2], keepdims = True)
                
        return tf.nn.relu(tf.reduce_sum(tf.multiply(alpha, self.feature_map), axis = [3]))

