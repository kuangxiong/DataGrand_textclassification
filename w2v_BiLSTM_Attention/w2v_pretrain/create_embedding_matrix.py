# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   create_embedding_matrix.py
@Time    :   2021/09/05 10:07:53
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''

import numpy as np 
import os 
import sys
from gensim.models import word2vec 
import pickle 

CUR_PATH = os.path.dirname(__file__)
print(CUR_PATH)

def load_wordVecter(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel 

def create_embedding_matrix(word2vec_path):
    """
    构造词嵌入矩阵和索引

    Args:
        word2vec_path ([str]): [词向量模型的路径]
    """
    model = load_wordVecter(word2vec_path)
    word2id = {"PAD": 0}
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    embedding_matrix = np.zeros((len(model.wv.vocab.items()) +1, model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2id[word] = i + 1
        embedding_matrix[i+1] = vocab_list[i][1]
    return word2id, embedding_matrix 


if __name__=='__main__':
    word2id, matrix = create_embedding_matrix(os.path.join(CUR_PATH, "word2vec.model"))
    file = open(os.path.join(CUR_PATH, "word2map.pkl"), "wb")
    pickle.dump(word2id, file)
    np.save(os.path.join(CUR_PATH, "w2v_matrix.npy"), matrix)

