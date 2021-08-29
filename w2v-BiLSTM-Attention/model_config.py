import os  
from ..config import GlobalData

class ModelConfig(GlobalData):
    """
    设置模型的超参数

    Args:
        GlobalData: 全局文件路径
    """
    # def __init__(self, hidden_size, max_length, embedding_size):
    self.hidden_size = hidden_size 
    self.max_length = 100
    self.embeddings_size = 512
    self.dropout = 0.5 
    self.num_epochs = 10 
    self.batch_size = 32 
    self.learning_rate = 0.1
