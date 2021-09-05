# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   data_utils.py
@Time    :   2021/08/30 18:49:23
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import tensorflow as tf           
from tensorflow import keras    

from w2v_config import w2v_Config, BASE_PATH   
import pickle 
import os 


def data_load(w2v_config):
    """
    用于读取训练数据和测试数据

    Args:
        w2v_config([class]):[w2v_config 参数类]
        train_file([str]):[训练数据]
        test_file([str]):[测试文件]
    """
    train_data, test_data = [], []
    
    