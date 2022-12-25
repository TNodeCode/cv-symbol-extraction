import torch
import torch.nn.functional as F


class EncoderRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=3, dropout=0.2, bidirectional=True, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(embedding_dim + 4, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, coordinates, hidden):
        # First run the input sequences through an embedding layer
        x = self.dropout(self.embedding(x))
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Concatenate embeddings with coordinates
        x = torch.cat([x, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers=3, dropout=0.2, bidirectional=True, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim + 0, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, hidden):
        # First run the input sequences through an embedding layer
        x = self.dropout(self.embedding(x))
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
