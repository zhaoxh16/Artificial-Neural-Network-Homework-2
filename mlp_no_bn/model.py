# -*- coding: utf-8 -*-

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


class Model:
    def __init__(self,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.hidden_layer_size = 32*16
        self.x_ = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
        self.y_ = tf.placeholder(tf.int32, [None])

        # TODO:  fill the blank of the arguments
        self.loss, self.pred, self.acc = self.forward(True, reuse=False)
        self.loss_val, self.pred_val, self.acc_val = self.forward(False, reuse=True)

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()

        # TODO:  maybe you need to update the parameter of batch_normalization?
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step, var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def forward(self, is_train, reuse=None):
        with tf.variable_scope("model", reuse=reuse):
            # TODO:  implement input -- Linear -- BN -- ReLU -- Dropout -- Linear -- loss
            #        the 10-class prediction output is named as "logits"
            # Your Linear Layer
            linear_1 = tf.layers.Dense(units=self.hidden_layer_size, name="Linear_1", trainable=is_train, _reuse=reuse)
            linear_1_out = linear_1(self.x_)
            # Your BN Layer: use batch_normalization_layer function
            # bn_1_out = batch_normalization_layer(linear_1_out, is_train=is_train)
            # Your Relu Layer
            relu_1_out = tf.nn.relu(linear_1_out, name='Relu_1')
            # Your Dropout Layer: use dropout_layer function
            # dropout_1_out = dropout_layer(relu_1_out, FLAGS.drop_rate, is_train=is_train)
            # Your Linear Layer
            linear_2 = tf.layers.Dense(units=10, name="Linear_2", trainable=is_train, _reuse=reuse)
            logits = linear_2(relu_1_out)
            # logits = tf.Variable(tf.constant(0.0, shape=[100, 10]))  # deleted this line after you implement above layers

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=logits))
        pred = tf.argmax(logits, 1)  # Calculate the prediction result
        correct_pred = tf.equal(tf.cast(pred, tf.int32), self.y_)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        return loss, pred, acc


def batch_normalization_layer(incoming, is_train=True):
    # TODO: implement the batch normalization function and applied it on fully-connected layers
    # NOTE:  If isTrain is True, you should use mu and sigma calculated based on mini-batch
    #       If isTrain is False, you must use mu and sigma estimated from training data
    bn_1 = tf.layers.BatchNormalization(name="BN_1", trainable=is_train, _reuse=(not is_train))
    return bn_1(incoming)


def dropout_layer(incoming, drop_rate, is_train=True):
    # TODO: implement the dropout function and applied it on fully-connected layers
    # Note: When drop_rate=0, it means drop no values
    #       If isTrain is True, you should randomly drop some values, and scale the others by 1 / (1 - drop_rate)
    #       If isTrain is False, remain all values not changed
    if is_train:
        dropout_1 = tf.layers.Dropout(rate=FLAGS.drop_rate, name="Dropout_1")
        dropout_1_out = dropout_1(incoming)
    else:
        dropout_1_out = incoming
    return dropout_1_out
