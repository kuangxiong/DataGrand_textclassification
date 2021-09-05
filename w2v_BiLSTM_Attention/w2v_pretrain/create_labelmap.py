# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   create_word2id.py
@Time    :   2021/08/31 23:13:44
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import os     
import sys 
import pandas as pd 
import pickle
from loguru import  logger
from gensim import corpora, models
from w2v_BiLSTM_Attention.model_config import ModelConfig

from config import GlobalData
logger = GlobalData.cust_logger

@logger.catch 
def labelmap(GlobalData):
    """
    数据label和id 之间的映射

    Args:
        GlobalData ([class]): [全局文件路径和参数配置]
    """
    logger.info('read train file')
    train_label = pd.read_csv(GlobalData.train_file)
    # 获取所有的label集合
    all_label = train_label['label'].unique()
    label2id, id2label = {}, {}
    for i in range(len(all_label)):
        label2id[all_label[i]] = i+1
        id2label[i+1] = all_label[i]
    label_map = [label2id, id2label]
    tmpfile = open(os.path.join(GlobalData.w2v_path, 'labelmap.pkl'), 'wb')
    pickle.dump(label_map, tmpfile)
    logger.info('save label map file')
    tmpfile.close()