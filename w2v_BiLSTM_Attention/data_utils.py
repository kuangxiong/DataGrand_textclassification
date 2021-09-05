# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   data_utils.py
@Time    :   2021/09/05 22:57:49
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import tensorflow as tf 
import numpy as np 
import os 
import sys
import pickle 
import pandas as pd 
from w2v_bilstm_attention.model_config import ModelConfig
from config import GlobalData 

logger = GlobalData.cust_logger

@logger.catch
def data_load(GlobalData):
    """
    读取训练集和测试集合

    Args:
        GlobalData ([class]): [全局文件路径配置]

    Returns:
        [list]: [预处理后的训练集和测试集合]
    """
    labelmap = pickle.load("w2v_pretrain/labelmap.pkl")    
    train_data = pd.read_csv(GlobalData.train_file)
    test_data = pd.read_csv(GlobalData.test_file)
    train_dataset, test_dataset = [], []
    for index, row in train_data.iterrows():
        single_sentence = row['text'].split()
        label = labelmap[row['label']]
        train_data.append([single_sentence, label])

    for index, row in test_data.iterrows():
        single_sentence = row['text'].split()
        test_data.append(single_sentence)
    return train_data, test_data

def get_vocab(word_dict_path):
    """
    获取词向量及词对应的ID编号

    Args:
        word_dict_path ([type]): [description]

    Returns:
        [obj]: [w2v 对象， 词向量模型]
    """
    word_dict_file = open(word_dict_file, "rb")
    word_dict = pickle.load(word_dict_file)
    return word_dict

def load_model_dataset(GlobalData):
