import torch
import torch.nn.functional as F


class EncoderRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers=1, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        # First run the input sequences through an embedding layer
        x = self.embedding(input)
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Now we need to run the embeddings through the LSTM layer
        x, hidden = self.lstm(x, hidden)
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        x = self.fc(x)
        # Finally map the outputs of the LSTM layer to a probability distribution
        x = self.softmax(x)
        # Return the prediction and the hidden state of the decoder
        return x, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
