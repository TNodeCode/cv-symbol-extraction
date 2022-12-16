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
        print("Input SHAPE", input.shape)
        embedded = self.embedding(input).view(1, 1, -1)
        print("Embedding SHAPE", embedded.shape)
        print("Hidden SHAPE", hidden.shape)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, device='cpu'):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
