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
from w2v_BiLSTM_Attention.model_config import ModelConfig
import pickle
from loguru import  logger
from gensim import corpora, models

@logger.catch 
def labelmap(GlobalData):
    """
    数据label和id 之间的映射

    Args:
        GlobalData ([class]): [全局文件路径和参数配置]
    """
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
    tmpfile.close()

@logger.catch 
def wordmap(GlobalData):
    """
    构造词和id之间的映射

    Args:
        GlobalData ([class]): [文件的全局路径配置文件]
    """
    train_data = pd.read_csv(GlobalData.train_file)
    test_data = pd.read_csv(GlobalData.test_file)

    all_text = pd.concat([train_data['text'], test_data['text']])
    print(all_text[0])
    all_word_list = []
    for i in range(len(all_text)):
        all_word_list.append(str(all_text.iloc[i]).split())
    word2map = corpora.Dictionary(all_word_list)
    word2map.save(os.path.join(GlobalData.w2v_path, 'wordmap.pkl'))

wordmap(ModelConfig)
# labelmap(ModelConfig)