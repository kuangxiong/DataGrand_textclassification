# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   w2v_config.py
@Time    :   2021/08/30 18:42:43
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import os 
import sys     

from w2v_BiLSTM_Attention.model_config import ModelConfig

class w2v_Config(ModelConfig):
    """
    word2vec 模型的参数设置

    Args:
        GlobalData（class): 一些文件全局路径
    """
    embedding_file = os.path.join(BASE_PATH, "model_data/w2v_matrix.npy")
    save_path = os.path.join(BASE_PATH, "model_data")