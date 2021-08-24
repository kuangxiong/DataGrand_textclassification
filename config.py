# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   config.py
@Time    :   2021/08/24 23:55:36
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''
import os 
import sys 

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

class GlobalData(object):
    """
    全局变量设置
    """
    train_file = os.path.join(BASE_PATH, "data_source/datagrand_2021_train.csv") 
    test_file = os.path.join(BASE_PATH, "data_source/datagrand_2021_test.csv")