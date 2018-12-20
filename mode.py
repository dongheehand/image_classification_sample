import numpy as np
from util import *
import tensorflow as tf
import os


def train(sess, saver, model, args):
    
    tr_img = read_all_images(args.tr_img)
    tr_img = data_aug(tr_img, args.aug_width, args.aug_height)
    tr_label = read_labels(args.tr_label, True)
    tr_num = tr_img.shape[0]
    tr_step = tr_num // args.batch_size
    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_path, sess.graph)
    txt_file = open(os.path.join(args.log_path, 'log.txt'), 'w')
    
    if args.test_with_train:

        te_img = read_all_images(args.te_img)
        te_label = read_labels(args.te_label)
        
        _, _, val_img, val_label = split_data(te_img, te_label)
        
        val_num = val_img.shape[0]
        val_step = val_num // args.test_batch
    
    best_val_acc = -1
    prev_val_acc = -1
    
    for i in range(args.max_epoch):
        random_index = np.random.permutation(tr_num)
        tr_loss_list = []
        
        for k in range(tr_step):
            tr_img_batch = train_batch_gen(tr_img, args, random_index, k)
            _, l, summary = sess.run([model.train, model.loss, merged], feed_dict = {model.global_step : i, model.phase : True,
                                                                   model.x : tr_img_batch, model.y : tr_label[random_index[args.batch_size * k : args.batch_size * (k+1)]]})
                
            tr_loss_list.append(l)
            
            if (i * tr_step + k) % args.log_freq:
                train_writer.add_summary(summary, i * tr_step + k)
            
        if i % args.model_save_freq == 0 :
            saver.save(sess, os.path.join(args.model_path, 'model_%04d'%(i)))
        
        tr_list = []
        for k_ in range(tr_step):
            index = np.arange(tr_num)
            tr_img_batch = train_batch_gen(tr_img, args, index, k_)
            tr_acc, _ = sess.run(model.acc, feed_dict = {model.phase : False, model.x : tr_img_batch, 
                                                      model.y : tr_label[index[args.batch_size * k_ : args.batch_size * (k_ + 1)]]})
            tr_list.append(tr_acc)
            
        tr_acc_ = sum(tr_list) / len(tr_list)
        loss_ = sum(tr_loss_list) / len(tr_loss_list)
        print('%04d-th epoch \t train_accuray : %04f \t train_loss : %04f' %(i, tr_acc_, loss_))            
        txt_file.write('%04d-th epoch \t train_accuray : %04f \t train_loss : %04f \t' %(i, tr_acc_, loss_))
        
        
        if args.test_with_train:
        
            val_acc_list = [] 
            for t in range(val_step):
                
                val_acc, _ = sess.run(model.acc, feed_dict = {model.phase : False, model.x : val_img[args.test_batch * t : args.test_batch * (t+1)],
                                                          model.y : val_label[args.test_batch * t : args.test_batch * (t+1)]})
                val_acc_list.append(val_acc)
            
            avg_val_acc = np.mean(np.asarray(val_acc_list))
            print('%04d-th epoch \t val_accuray : %04f' %(i, avg_val_acc))
            txt_file.write('val_accuray : %04f' %(avg_val_acc))
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                saver.save(sess, os.path.join(args.model_path, 'model_best_on_val'))
        
        txt_file.write('\n')
        
    saver.save(sess, os.path.join(args.model_path, 'model_final'))
    txt_file.close()
    
def test(sess, saver, model, args):
    
    saver.restore(sess, args.pre_trained_model)
    print('Saved model is loaded for test!')
    print('Model path is %s'%args.pre_trained_model)
    print('Number of parameters : %d' % model.model_parameter())
    
    te_img = read_all_images(args.te_img)
    te_label = read_labels(args.te_label)    

    te_img, te_label, val_img, val_label = split_data(te_img, te_label)

    te_num = te_img.shape[0]
    val_num = val_img.shape[0]
    te_step = te_num // args.test_batch
    val_step = val_num // args.test_batch

    te_acc_list = []

    for t in range(te_step):
        te_acc, _ = sess.run(model.acc, feed_dict = {model.phase : False, model.x : te_img[args.test_batch * t : args.test_batch * (t+1)],
                                                 model.y : te_label[args.test_batch * t : args.test_batch * (t+1)]})
        te_acc_list.append(te_acc)

    print('test_accuray : %04f' %(np.mean(np.asarray(te_acc_list))))

    val_acc_list = []

    for t in range(val_step):
        val_acc, _ = sess.run(model.acc, feed_dict = {model.phase : False, model.x : val_img[args.test_batch * t : args.test_batch *(t+1)],
                                                  model.y : val_label[args.test_batch * t : args.test_batch * (t+1)]})
        val_acc_list.append(val_acc)

    print('val_accuray : %04f' %(np.mean(np.asarray(val_acc_list))))

def grad_cam(sess, saver, model, args):
    
    saver.restore(sess, args.pre_trained_model)
    
    print('Saved model is loaded for grad cam!')
    print('Model path is %s'%args.pre_trained_model)
    print('Number of parameters : %d' % model.model_parameter())
    
    te_img = read_all_images(args.te_img)
    te_label = read_labels(args.te_label)    

    te_img, te_label, val_img, val_label = split_data(te_img, te_label)

    te_num = te_img.shape[0]
    val_num = val_img.shape[0]
    te_step = te_num // args.test_batch
    val_step = val_num // args.test_batch
    
    for t in range(te_step):
        result, cam_img = sess.run([model.acc, model.grad_cam()], 
                                        feed_dict = {model.phase : False, 
                                                     model.x : te_img[args.test_batch * t : args.test_batch * (t+1)], 
                                                     model.y : te_label[args.test_batch * t : args.test_batch * (t+1)]})
        is_true = result[1]
        for k in range(args.test_batch):
            cam = save_cam(cam_img[k], te_img[args.test_batch * t + k])
            if is_true[k]:
                cv2.imwrite('./result/true/%d.png'%(args.test_batch * t + k), cam)
            else:
                cv2.imwrite('./result/false/%d.png'%(args.test_batch * t + k), cam)

