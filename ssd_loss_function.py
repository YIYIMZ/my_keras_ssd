# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:24:51 2018

@author: yy
"""

from __future__ import division
import tensorflow as tf

class SSDLoss:

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred)**2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
       
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0] 
        n_boxes = tf.shape(y_pred)[1] 

        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12]))            # 输出: (batch_size, n_boxes)
        localization_loss = tf.to_float(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]))    # 输出: (batch_size, n_boxes)

        # 计算分类loss

        negatives = y_true[:,:,0]                                                   # 输出: (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis=-1))          # 输出: (batch_size, n_boxes)

        n_positive = tf.reduce_sum(positives)

        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)    # 输出: (batch_size,)

        neg_class_loss_all = classification_loss * negatives    # 输出: (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32) 
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min), n_neg_losses)

        
        def f1():
            return tf.zeros([batch_size])
        
        def f2():
            
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])            # 输出: (batch_size * n_boxes,)
            
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False) 
            
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(neg_class_loss_all_1D))           # 输出: (batch_size * n_boxes,)
            negatives_keep = tf.to_float(tf.reshape(negatives_keep, [batch_size, n_boxes])) # 输出: (batch_size, n_boxes)
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)   # 输出: (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss    # 输出: (batch_size,)

        # 计算loc loss

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1) # 输出: (batch_size,)

        # 计算总 loss.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)
        
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss
