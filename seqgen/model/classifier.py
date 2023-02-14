import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    This module takes the outputs of a decoder and runs them through a classification network
    that consists of linear layers and a final softmax classification layer.
    """
    def __init__(
        self,
        trg_vocab_size,
        embedding_dim,
        softmax_dim=2,
    ):
        super(Classifier, self).__init__()
        
        # Hyperparameters
        self.trg_vocab_size = trg_vocab_size
        self.embedding_dim = embedding_dim
        
        # Layers
        self.fc = nn.Linear(embedding_dim, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=softmax_dim)
        
    def forward(self, x):
        return self.softmax(self.fc(x))