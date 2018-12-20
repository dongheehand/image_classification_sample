import tensorflow as tf
import numpy as np
from res_model import model
from mode import *
import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

parser.add_argument("--aug_width", type = int, default = 108)
parser.add_argument("--aug_height", type = int, default = 108)
parser.add_argument("--width", type = int, default = 96)
parser.add_argument("--height", type = int, default = 96)
parser.add_argument("--num_label", type = int, default = 10)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--max_epoch", type = int, default = 500)
parser.add_argument("--model_save_freq", type = int, default = 50)
parser.add_argument("--learning_rate", type = float, default = 5e-3)
parser.add_argument("--decay_step", type = int, default = 50)
parser.add_argument("--decay_rate", type = float, default = 0.8)
parser.add_argument("--test_batch", type = int, default = 20)
parser.add_argument("--log_freq", type = int, default = 100)
parser.add_argument("--model_path", type = str, default = './model')
parser.add_argument("--log_path", type = str, default = './log')
parser.add_argument("--test_with_train", type = str2bool, default = True)
parser.add_argument("--mode", type = str, default = 'train')
parser.add_argument("--pre_trained_model", type = str, default = './model/model_final')
parser.add_argument("--tr_img", type = str, default = './stl10_binary/test_X.bin')
parser.add_argument("--tr_label", type = str, default = './stl10_binary/test_y.bin')
parser.add_argument("--te_img", type = str, default = './stl10_binary/train_X.bin')
parser.add_argument("--te_label", type = str, default = './stl10_binary/train_y.bin')


args = parser.parse_args()

model = model(args)
model.build_model()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep = None)


if args.mode == 'train':
    train(sess, saver, model, args)

elif args.mode == 'test':
    test(sess, saver, model, args)

elif args.mode == 'grad_cam':
    grad_cam(sess, saver, model, args)

else:
    print('mode option is wrong!')

