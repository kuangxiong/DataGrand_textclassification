# -*- encoding: utf-8 -*-
'''
--------------------------------------------------------------------
@File    :   w2v_gensim.py
@Time    :   2021/08/31 07:48:55
@Author  :   kuangxiong 
@Version :   1.0
@Email :   kuangxiong1993@163.com
--------------------------------------------------------------------
'''
import pandas as pd 
import os
from gensim.models import word2vec
from gensim import utils
import multiprocessing
from w2v_bilstm_attention.model_config import ModelConfig


def train_wordVectors(sentences, embedding_size=128, window=5, min_count=5):
    """

    :param sentences: sentences可以是LineSentence或者PathLineSentences读取的文件对象，也可以是
                    The `sentences` iterable can be simply a list of lists of tokens,
                    如lists=[['我','是','中国','人'],['我','的','家乡','在','广东']]
    :param embedding_size: 词嵌入大小
    :param window: 窗口
    :param min_count:Ignores all words with total frequency lower than this.
    :return: w2vModel
    """
    w2vModel = word2vec.Word2Vec(
        sentences,
        size=embedding_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
    )
    return w2vModel


def save_wordVectors(w2vModel, word2vec_path):
    w2vModel.save(word2vec_path)


def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel


def train_grand_data(ModelConfig):
    """
    采用word2vec模型训练词向量

    Args:
        ModelConfig ([class]): [description]
    """
    train_data = pd.read_csv(ModelConfig.train_file)
    test_data = pd.read_csv(ModelConfig.test_file)
    all_data = pd.concat([train_data['text'], test_data['text']])
    sentences = []
    for i in range(len(all_data)):
        sentences.append(all_data.iloc[i].split())
    print(sentences[0])
    # sentences = word2vec.PathLineSentences(segment_dir)
    model = train_wordVectors(sentences, embedding_size=128, window=5, min_count=1)
    save_wordVectors(model, os.path.join(ModelConfig.w2v_path, "word2vec.model"))

def train_test():
    # [1]若只有一个文件，使用LineSentence读取文件
    segment_path='./data/segment/segment_0.txt'
    utils.to_unicode(segment_path, encoding="utf-8", errors="ignore")
    sentences = word2vec.LineSentence(segment_path)

    # [1]若存在多文件，使用PathLineSentences读取文件列表

#segment_dir = "./data/segment"
#    sentences = word2vec.PathLineSentences(segment_dir)

    # 简单的训练
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
#print(model.wv.similarity("沙瑞金", "高育良"))
    # print(model.wv.similarity('李达康'.encode('utf-8'), '王大路'.encode('utf-8')))

    # 一般训练，设置以下几个参数即可：
    word2vec_path = "./models/word2Vec.model"
    model2 = train_wordVectors(sentences, embedding_size=256, window=5, min_count=5)
    save_wordVectors(model2, word2vec_path)
    model2 = load_wordVectors(word2vec_path)
#print(model2.wv.similarity("沙瑞金", "高育良"))
#    print(model2.wv.similarity("后", "前"))
    print(model2.wv["我"])


if __name__ == "__main__":
    train_grand_data(ModelConfig)
    # segment_dir='data/THUCNews_segment'
    # out_word2vec_path='models/THUCNews_word2Vec/THUCNews_word2Vec_128.model'
    # # train_THUCNews(segment_dir, out_word2vec_path)
    #
    # w2vModel=load_wordVectors(out_word2vec_path)
    # word1='文化'
    # word2='紧急'
    # vecor1=w2vModel[word1]
    # vecor2=w2vModel[word2]
    # print(w2vModel.wv.similarity('你', '我'))
