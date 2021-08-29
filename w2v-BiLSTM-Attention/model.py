from model_config import ModelConfig
import os 
import tensorflow as tf
from tensorlow import keras 
from attention import Attention


def BiLSTM_Attention(ModelConfig):
    """
    定义BiLSTM模型的网络结构

    Args:
        ModelConfig (类): 描述模型网络结构的超参数
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(ModelConfig.n_vocab, ModelConfig.embedding_size))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2, 
             return_sequences=True, dropout=ModelConfig.dropout)))
    model.add(Attention(name='attention_weight'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model




def W2v_BiLSTM_Attention(ModelConfig):
    """
    定义w2v-BiLSTM模型的网络结构

    Args:
        ModelConfig (类): 描述模型网络结构的超参数
    """
    model = keras.models.Sequential()
    # 此处 夹在w2v 模块
    model.add(keras.layers.Embedding(ModelConfig.n_vocab, ModelConfig.embedding_size))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(ModelConfig.hidden_size//2, 
             return_sequences=True, dropout=ModelConfig.dropout)))
    model.add(Attention(name='attention_weight'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    return model