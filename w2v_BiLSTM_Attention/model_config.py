import sys
print(sys.path)
import os  
import pickle

from config import GlobalData

class ModelConfig(GlobalData):
    """
    设置模型的超参数

    Args:
        GlobalData: 全局文件路径
    """
    # def __init__(self, hidden_size, max_length, embedding_size):
    w2v_path = "w2v_BiLSTM_Attention/w2v_pretrain"
    word_dict_path = os.path.join(os.path.dirname(__file__), "w2v_pretrain/word2map.pkl")
    __file = open(word_dict_path, 'rb')
    n_vocab = len(pickle.load(__file))+1
    hidden_size = 64
    max_length = 100
    embedding_size = 512
    dropout = 0.5 
    num_epochs = 10 
    batch_size = 32 
    learning_rate = 0.1

