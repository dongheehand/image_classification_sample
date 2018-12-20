import tensorflow as tf
import numpy as np
import os

class dataloader():
    
    def __init__(self, args):
        
        self.aug_width = args.aug_width
        self.aug_height = args.aug_height
        self.image_width = args.width
        self.image_height = args.height
        self.batch_size = args.batch_size
        self.test_batch = args.test_batch
        self.test_with_train = args.test_with_train
        self.mode = args.mode
        self.image = tf.placeholder(shape = [None, args.height, args.width, 3],dtype = tf.float32)
        self.label = tf.placeholder(shape = [None, args.label], dtype = tf.float32)
        
        
    def build_loader(self):
        
        if self.mode == 'train':
            
            tr_data = (self.image, self.label)
            self.tr_dataset = tf.data.Dataset.from_tensor_slices(tr_data)
            
            self.tr_dataset = self.tr_dataset.map(self.resize_aug, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.map(self.flip_aug, num_parallel_calls = 4).prefetch(32)
            self.tr_dataset = self.tr_dataset.shuffle(32)
            self.tr_dataset = self.tr_dataset.repeat()
            self.tr_dataset = self.tr_dataset.batch(self.batch_size)
            
            iterator = tf.data.Iterator.from_structure(self.tr_dataset.output_types, self.tr_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['tr_init'] = iterator.make_initializer(self.tr_dataset)
            
            if self.test_with_train:
                self.val_image = tf.placeholder(shape = [None, 96, 96, 3],dtype = tf.float32)
                self.val_label = tf.placeholder(shape = [None, 10], dtype = tf.float32)
                val_data = (self.val_image, self.val_label)
                self.val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
                self.val_dataset = self.val_dataset.batch(self.test_batch)
                self.init_op['val_init'] = iterator.make_initializer(self.val_dataset)
                
        elif self.mode == 'test':
            
            te_data = (self.image, self.label)
            self.te_dataset = tf.data.Dataset.from_tensor_slices(te_data)
            
            self.te_dataset = self.te_dataset.shuffle(32)
            self.te_dataset = self.te_dataset.batch(self.test_batch)
            
            iterator = tf.data.Iterator.from_structure(self.te_dataset.output_types, self.te_dataset.output_shapes)
            self.next_batch = iterator.get_next()
            self.init_op = {}
            self.init_op['te_init'] = iterator.make_initializer(self.te_dataset)            
        
    def resize_aug(self, image, label):
        
        image = tf.image.resize_images(image, (self.aug_height, self.aug_width), tf.image.ResizeMethod.BICUBIC)
        
        shape = tf.shape(image)
        ih = shape[0]
        iw = shape[1]
        
        ix = tf.random_uniform(shape = [1], minval = 0, maxval = iw - self.image_width + 1, dtype = tf.int32)[0]
        iy = tf.random_uniform(shape = [1], minval = 0, maxval = ih - self.image_height + 1, dtype = tf.int32)[0]
        
        image = image[iy:iy + self.image_height, ix:ix + self.image_width]
        
        return image, label
    
    def flip_aug(self, image, label):
        
        flip_rl = tf.random_uniform(shape = [1], minval = 0, maxval = 3, dtype = tf.int32)[0]

        rl = tf.equal(tf.mod(flip_rl, 2),0)
        
        image = tf.cond(rl, true_fn = lambda : tf.image.flip_left_right(image), false_fn = lambda : (image))
        
        return image, label

