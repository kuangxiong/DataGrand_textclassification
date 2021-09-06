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
# from loguru import logger
logger = GlobalData.cust_logger

# @logger.catch
def data_load(ModelConfig):
    """
    读取训练集和测试集合

    Args:
        GlobalData ([class]): [全局文件路径配置]

    Returns:
        [list, list]: [原始文本预处理后的训练集和测试集合]
    """
    pkl_file = open("w2v_bilstm_attention/w2v_pretrain/labelmap.pkl", 'rb')
    label2id, id2label = pickle.load(pkl_file)
 
    train_data = pd.read_csv(GlobalData.train_file)
    test_data = pd.read_csv(GlobalData.test_file)
    train_dataset, test_dataset = [], []
    for index, row in train_data.iterrows():
        single_sentence = row['text'].split()
        label = label2id[str(row['label'])]
        train_dataset.append([single_sentence, label])

    for index, row in test_data.iterrows():
        single_sentence = row['text'].split()
        test_dataset.append(single_sentence)
    return train_dataset, test_dataset

# @logger.catch
def get_vocab(word_dict_path):
    """
    获取词向量及词对应的ID编号

    Args:
        word_dict_path ([str]): [word2map.pkl 文件的路径]

    Returns:
        [obj]: [w2v 对象， 词向量模型]
    """
    word_dict_file = open(word_dict_path, "rb")
    word_dict = pickle.load(word_dict_file)
    return word_dict

# @logger.catch
def load_model_dataset(ModelConfig):
    """
    数据预处理，用于生成模型输入的数据结构

    Args:
        ModelConfig ([obj]): [模型参数和配置文件的路径]

    Returns:
        [list, list]: [预处理后的训练集和测试集]
    """
    train_text, test_text = data_load(ModelConfig)
    word_dict = get_vocab(ModelConfig.word_dict_path)
    train_dataset, test_dataset = [], []
    for i in range(len(train_text)):
        tmp_list = list(map(lambda x: word_dict[x], train_text[i][0]))
        train_dataset.append([tmp_list, train_text[i][1]])

    for i in range(len(test_text)):
        tmp_list = list(map(lambda x:word_dict[x], test_text[i]))
        test_dataset.append(tmp_list)
    logger.warn("数据预处理文件完成")
    return train_dataset, test_dataset