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
from ...config import GlobalData

def wordmap(GlobalData):
    """
    构造词和id之间的映射

    Args:
        GlobalData ([class]): [文件的全局路径配置文件]
    """
    print(1111, GlobalData.train_file)

wordmap(GlobalData)
