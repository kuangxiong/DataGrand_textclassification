import sys
print(sys.path)
import os  

from config import GlobalData

class ModelConfig(GlobalData):
    """
    设置模型的超参数

    Args:
        GlobalData: 全局文件路径
    """
    # def __init__(self, hidden_size, max_length, embedding_size):
    w2v_path = "w2v_BiLSTM_Attention/w2v_pretrain"
    hidden_size = 64
    max_length = 100
    embeddings_size = 512
    dropout = 0.5 
    num_epochs = 10 
    batch_size = 32 
    learning_rate = 0.1
print(ModelConfig)
