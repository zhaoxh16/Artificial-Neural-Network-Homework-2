# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
from model import Model
from load_data import load_cifar_4d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('AGG')

tf.app.flags.DEFINE_integer("batch_size", 100, "batch size for training")
tf.app.flags.DEFINE_integer("num_epochs", 20, "number of epochs")
tf.app.flags.DEFINE_float("drop_rate", 0.5, "drop out rate")
tf.app.flags.DEFINE_boolean("is_train", True, "False to inference")
tf.app.flags.DEFINE_string("data_dir", "../cifar-10_data", "data dir")
tf.app.flags.DEFINE_string("train_dir", "./train", "training dir")
tf.app.flags.DEFINE_integer("inference_version", 0, "the version for inference")
FLAGS = tf.app.flags.FLAGS


def shuffle(X, y, shuffle_parts):
    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def train_epoch(model, sess, X, y): # Training Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch}
        loss_, acc_, _ = sess.run([model.loss, model.acc, model.train_op], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def valid_epoch(model, sess, X, y): # Valid Process
    loss, acc = 0.0, 0.0
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = X[st:ed], y[st:ed]
        feed = {model.x_: X_batch, model.y_: y_batch}
        loss_, acc_ = sess.run([model.loss_val, model.acc_val], feed)
        loss += loss_
        acc += acc_
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    acc /= times
    return acc, loss


def inference(model, sess, X):  # Test Process
    return sess.run([model.pred_val], {model.x_: X})[0]


with tf.Session() as sess:
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    if FLAGS.is_train:
        X_train, X_test, y_train, y_test = load_cifar_4d(FLAGS.data_dir)
        X_val, y_val = X_train[40000:], y_train[40000:]
        X_train, y_train = X_train[:40000], y_train[:40000]
        cnn_model = Model()

        writer = tf.summary.FileWriter(FLAGS.train_dir)
        writer.add_graph(tf.get_default_graph())
        writer.flush()

        # if tf.train.get_checkpoint_state(FLAGS.train_dir):
        #     cnn_model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        # else:
        tf.global_variables_initializer().run()

        pre_losses = [1e18] * 3
        best_val_acc = 0.0
        best_test_acc = 0.0
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        train_acc_list = []
        val_acc_list = []
        test_acc_list = []
        for epoch in range(FLAGS.num_epochs):
            start_time = time.time()
            train_acc, train_loss = train_epoch(cnn_model, sess, X_train, y_train)
            X_train, y_train = shuffle(X_train, y_train, 1)

            val_acc, val_loss = valid_epoch(cnn_model, sess, X_val, y_val)
            test_acc, test_loss = valid_epoch(cnn_model, sess, X_test, y_test)

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1
                cnn_model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=cnn_model.global_step)

            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch + 1) + " of " + str(FLAGS.num_epochs) + " took " + str(epoch_time) + "s")
            print("  learning rate:                 " + str(cnn_model.learning_rate.eval()))
            print("  training loss:                 " + str(train_loss))
            print("  training accuracy:             " + str(train_acc))
            print("  validation loss:               " + str(val_loss))
            print("  validation accuracy:           " + str(val_acc))
            print("  best epoch:                    " + str(best_epoch))
            print("  best validation accuracy:      " + str(best_val_acc))
            print("  test loss:                     " + str(test_loss))
            print("  test accuracy:                 " + str(test_acc))
            print("  best test accuracy             " + str(best_test_acc))
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            test_loss_list.append(test_loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

            if train_loss > max(pre_losses):
                sess.run(cnn_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [train_loss]

        x = range(FLAGS.num_epochs)
        plt.cla()
        plt.plot(x, train_loss_list, label="train loss")
        plt.plot(x, test_loss_list, label="test loss")
        plt.plot(x, val_loss_list, label="val loss")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(FLAGS.train_dir, "loss.png")))
        plt.cla()
        plt.plot(x, train_acc_list, label="train acc")
        plt.plot(x, test_acc_list, label="test acc")
        plt.plot(x, val_acc_list, label="val acc")
        plt.legend()
        plt.savefig(os.path.join(os.path.join(FLAGS.train_dir, "acc.png")))

        with open(os.path.join(FLAGS.train_dir, "info.txt"), 'w') as f:
            info = {
                "batch_size": FLAGS.batch_size,
                "num_epochs": FLAGS.num_epochs,
                "drop_rate": FLAGS.drop_rate,
                "train_dir": FLAGS.train_dir,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_acc
            }
            f.write(str(info))
    else:
        cnn_model = Model()
        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        cnn_model.saver.restore(sess, model_path)
        X_train, X_test, y_train, y_test = load_cifar_4d(FLAGS.data_dir)

        count = 0
        for i in range(len(X_test)):
            test_image = X_test[i].reshape((1, 32, 32, 3))
            result = inference(cnn_model, sess, test_image)[0]
            if result == y_test[i]:
                count += 1
        print("test accuracy: {}".format(float(count) / len(X_test)))
