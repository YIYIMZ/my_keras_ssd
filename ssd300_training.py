# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:59:55 2018

@author: yy
"""

from keras.optimizers import Adam, SGD
from keras import backend as K

from ssd300 import SSD300
from keras_loss_function.keras_ssd_loss import SSDLoss

img_height = 300 
img_width = 300 
img_channels = 3 
n_classes = 20

K.clear_session() 

# 1: 创建模型
model = SSD300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training')

# 2: 加载VGG16模型权重
weights_path = 'vgg/VGG_ILSVRC_16_layers_fc_reduced.h5'

model.load_weights(weights_path, by_name=True)

# 3: 优化器及loss

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)
model_path = 'path/to/trained/model.h5'

#数据加载























