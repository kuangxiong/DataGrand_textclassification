# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   model_train.py
@Time    :   2021/09/05 22:25:56
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import numpy as np 
import pickle 
import time
import tensorflow as tf 
from tensorflow import keras

from w2v_bilstm_attention.model_config import ModelConfig
from w2v_bilstm_attention.model_backbone import bilstm_attention
from w2v_bilstm_attention.data_utils import load_model_dataset
from config import GlobalData 

logger = GlobalData.cust_logger 

if __name__=="__main__":
    start_time = time.time()
    train_data, test_data = load_model_dataset(ModelConfig)
    logger.debug("load train & test data")
    model = bilstm_attention(ModelConfig)
    adam = tf.keras.optimizers.Adam(ModelConfig.learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=adam,
        metrics=['accuracy']
    )
    logger.info("model config")
    print(model.summary())






