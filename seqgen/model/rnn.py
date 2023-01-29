import torch
import torch.nn.functional as F
import numpy as np
from seqgen.model.embedding import *
from seqgen.model.attention import *
from seqgen.model.classifier import *

    
class CellType:
    RNN = "rnn"
    GRU = "gru"
    LSTM = "lstm"
    
    @staticmethod
    def rnn_layer(cell_type):
        cell_types = {
            "rnn": torch.nn.RNN,
            "gru": torch.nn.GRU,
            "lstm": torch.nn.LSTM,
        }
        return cell_types[cell_type]


class RecurrentEncoder(torch.nn.Module):
    def __init__(self, cell_type, vocab_size, embedding_dim, hidden_size, max_length, embedding_type=EmbeddingType.COORDS_DIRECT, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(RecurrentEncoder, self).__init__()
        self.device = device
        self.cell_type = cell_type
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        ### Layers ###
        self.embedding = EmbeddingType.embedding_layer(embedding_type)(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_prob=dropout,
            max_length=max_length,
            device=device            
        )
        self.rnn = CellType.rnn_layer(cell_type)(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x, coordinates, hidden):
        # First run the input sequences through an embedding layer
        embedded = self.embedding(x=x, coordinates=coordinates)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)


class EncoderGRUPosEnc(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(EncoderGRUPosEnc, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        ### Layers ###
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        embedded = self.dropout(self.embedding(x) + position)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)


class EncoderGRUCoords(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(EncoderGRUCoords, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        ### Layers ###
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.GRU(embedding_dim+4, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        embedded = self.dropout(self.embedding(x))
        # Concatenate embeddings with coordinates
        embedded = torch.cat([embedded, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Add positional encoding to the embedding
        if self.pos_encoding:
            x = x + position
        # Add dropout to prevent overfitting
        x = self.dropout(x)
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Concatenate embeddings with coordinates
        #x = torch.cat([x, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        x, hidden = self.lstm(x, hidden)
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        x = self.fc(x)
        # Finally map the outputs of the LSTM layer to a probability distribution
        x = self.softmax(x)
        # Return the prediction and the hidden state of the decoder
        return x, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)

    
class DecoderGRU(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(DecoderGRU, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.gru = torch.nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Add positional encoding to the embedding
        if self.pos_encoding:
            x = x + position
        # Add dropout to prevent overfitting
        x = self.dropout(x)
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Concatenate embeddings with coordinates
        #x = torch.cat([x, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        x, hidden = self.gru(x, hidden)
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        x = self.fc(x)
        # Finally map the outputs of the LSTM layer to a probability distribution
        x = self.softmax(x)
        # Return the prediction and the hidden state of the decoder
        return x, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)

    
class RecurrentAttentionDecoder(torch.nn.Module):
    def __init__(self, cell_type, embedding_type, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=False, pos_encoding=False, attention_type=AttentionType.DOT, device='cpu'):
        super(RecurrentAttentionDecoder, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.cell_type = cell_type
        self.max_length = max_length
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi_factor = 2 if bidirectional else 1
        self.pos_encoding = pos_encoding
        self.attention_type = attention_type
        
        # Layers
        self.embedding = EmbeddingType.embedding_layer(embedding_type)(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout_prob=dropout,
            max_length=max_length,
            device=device            
        )
        #self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        #self.attn_hn = torch.nn.Linear(hidden_size*(self.bi_factor*2), vocab_size)
        self.rnn = CellType.rnn_layer(cell_type)(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.attn = AttentionType.attention_layer(attention_type)(
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            max_length=max_length
        )
        self.cls = Classifier(
            trg_vocab_size=vocab_size,
            embedding_dim=hidden_size*(self.bi_factor*2),
            softmax_dim=1,
        )
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, positions, annotations, hidden):
        # First run the input sequences together with the positions through an embedding layer
        embedded = self.embedding(x=x, pos=positions)
        # Now we need to run the embeddings through the LSTM layer
        rnn_output, hidden_new = self.rnn(embedded, hidden)
        # Compute attention and context vector
        context_vector, attention = self.attn(
            hidden_new[0] if self.cell_type == CellType.LSTM else hidden_new,
            annotations
        )
        # Concatenate ocntext vector and output of RNN layer
        #output = self.attn_hn(torch.cat([rnn_output.squeeze(), context_vector.squeeze()], dim=1))
        # Finally map the outputs of the LSTM layer to a probability distribution
        #output = self.softmax(output)
        output = self.cls(torch.cat([rnn_output.squeeze(), context_vector.squeeze()], dim=1))
        # Return the prediction and the hidden state of the decoder
        return output, hidden_new, attention

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)
