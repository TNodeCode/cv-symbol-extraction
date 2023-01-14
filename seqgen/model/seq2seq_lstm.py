import torch
import torch.nn.functional as F
import numpy as np

LOGGING=False
def _log(*args):
    if LOGGING:
        print(*args)
        

# @see: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/#:~:text=What%20Is%20Positional%20Encoding%3F,item%27s%20position%20in%20transformer%20models.
# @see: https://kikaben.com/transformers-positional-encoding/
def get_position_encoding(seq_len, d, n=10000, device='cpu'):
    P = torch.zeros((seq_len, d)).to(device)
    for k in range(seq_len):
        for i in torch.arange(int(d/2)):
            denominator = n ** (2*i/d)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P

def get_coordinate_encoding(coordinates, d, n=10000, device='cpu'):
    batch_size = coordinates.shape[0]
    seq_len = coordinates.shape[1]
    P = torch.zeros((batch_size, seq_len, d)).to(device)
    for k in range(seq_len):
        for i in torch.arange(int(d/8)):
            denominator = n ** (2*i/d)
            P[:, k, 8*i+0] = torch.sin(coordinates[:, k, 0]/denominator)
            P[:, k, 8*i+1] = torch.cos(coordinates[:, k, 0]/denominator)
            P[:, k, 8*i+2] = torch.sin(coordinates[:, k, 1]/denominator)
            P[:, k, 8*i+3] = torch.cos(coordinates[:, k, 1]/denominator)
            P[:, k, 8*i+4] = torch.sin(coordinates[:, k, 2]/denominator)
            P[:, k, 8*i+5] = torch.cos(coordinates[:, k, 2]/denominator)
            P[:, k, 8*i+6] = torch.sin(coordinates[:, k, 3]/denominator)
            P[:, k, 8*i+7] = torch.cos(coordinates[:, k, 3]/denominator)
    return P


def repeat_hidden_state(hn, max_length):
    return hn.reshape(1, hn.size(0), hn.size(1)).repeat(max_length, 1, 1).permute(1,0,2)


def concat_hidden_states(hn):
    """
    Concatenate hidden state vectors from multiple hidden layers
    :parameter hn: Hidden state tensor of shape (num_layers, batch_size, hidden_size)
    :return: Concatenated state tensor of shape (batch_size, hidden_size*num_layers)
    """
    return hn.permute(1,0,2).reshape(hn.size(1), -1)


def concat_and_repeat_hidden_state(hn, max_length):
    hn_cat = concat_hidden_states(hn)
    return repeat_hidden_state(hn_cat, max_length=max_length)


def combine_encoder_annotations_and_hidden_state(hn, annotations):
    max_length = annotations.size(1)
    hn_rep = concat_and_repeat_hidden_state(hn, max_length=max_length)
    return torch.cat([hn_rep, annotations], dim=2)


class EncoderRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Add positional encoding to the embedding
        if self.pos_encoding:
            x = x + position        
        x = self.dropout(x)
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Concatenate embeddings with coordinates
        #x = torch.cat([x, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)


class EncoderGRU(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, max_length, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(EncoderGRU, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_length = max_length
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, coordinates, position, hidden):
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Add positional encoding to the embedding
        if self.pos_encoding:
            x = x + position        
        x = self.dropout(x)
        # Next pass the embeddings to an activation function
        x = F.relu(x)
        # Concatenate embeddings with coordinates
        #x = torch.cat([x, coordinates.unsqueeze(dim=1)], dim=2)
        # Now we need to run the embeddings through the LSTM layer
        output, hidden = self.gru(x, hidden)
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

    
class DecoderLSTMAttention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=False, pos_encoding=False, device='cpu'):
        super(DecoderLSTMAttention, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding
        use_last_n_states=num_layers
        self.attn = AdditiveAttention(hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, max_length=max_length, use_last_n_states=use_last_n_states)
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if bidirectional:
            self.context_dim = 2*use_last_n_states*hidden_size
        else:
            self.context_dim = use_last_n_states*hidden_size
        self.attn_hn = torch.nn.Linear(hidden_size*2, hidden_size)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, annotations, position, hidden):
        n_hidden_states = 2*self.num_layers if self.bidirectional else self.num_layers
        context_vector, attention = self.attn(hidden[0].detach(), annotations.detach())
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Concatenate embeddings with context vector
        #x = torch.cat([x, context_vector], dim=2)
        hn_attn = F.tanh(self.attn_hn(torch.cat([context_vector.squeeze(), concat_hidden_states(hidden[0])], dim=1).reshape(hidden[0].size(0), hidden[0].size(1), -1)))
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
        x, hidden = self.lstm(x, (hn_attn, hidden[1]))
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        x = self.fc(x)
        # Finally map the outputs of the LSTM layer to a probability distribution
        x = self.softmax(x)
        # Return the prediction and the hidden state of the decoder
        return x, hidden, attention

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)

    
class DecoderGRUAttention(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=False, pos_encoding=False, device='cpu'):
        super(DecoderGRUAttention, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding
        use_last_n_states=num_layers
        self.attn = AdditiveAttention(hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, max_length=max_length, use_last_n_states=use_last_n_states)
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if bidirectional:
            self.context_dim = 2*use_last_n_states*hidden_size
        else:
            self.context_dim = use_last_n_states*hidden_size
        self.attn_hn = torch.nn.Linear(hidden_size*2, hidden_size)
        self.gru = torch.nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, annotations, position, hidden):
        n_hidden_states = 2*self.num_layers if self.bidirectional else self.num_layers
        # First run the input sequences through an embedding layer
        x = self.embedding(x)
        # Compute attention and context vector
        context_vector, attention = self.attn(hidden, annotations.detach())
        hn_attn = F.tanh(self.attn_hn(torch.cat([context_vector.squeeze(), concat_hidden_states(hidden)], dim=1).reshape(hidden.size(0), hidden.size(1), -1)))
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
        x, hidden = self.gru(x, hn_attn)
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        x = self.fc(x)
        # Finally map the outputs of the LSTM layer to a probability distribution
        x = self.softmax(x)
        # Return the prediction and the hidden state of the decoder
        return x, hidden, attention

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)
    

    
class DecoderGRUAttention2(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length, num_layers=3, dropout=0.1, bidirectional=False, pos_encoding=False, device='cpu'):
        super(DecoderGRUAttention2, self).__init__()
        self.device = device
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding
        use_last_n_states=num_layers
        self.attn = AdditiveAttention(hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, max_length=max_length, use_last_n_states=use_last_n_states)
        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if bidirectional:
            self.context_dim = 2*use_last_n_states*hidden_size
        else:
            self.context_dim = use_last_n_states*hidden_size
        self.attn_hn = torch.nn.Linear(hidden_size*(num_layers+1), vocab_size)
        self.gru = torch.nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, coordinates, annotations, position, hidden):
        n_hidden_states = 2*self.num_layers if self.bidirectional else self.num_layers
        # First run the input sequences through an embedding layer
        embedded = self.embedding(x)
        # Add dropout to prevent overfitting
        embedded = self.dropout(embedded)
        # Next pass the embeddings to an activation function
        #embedded = F.relu(embedded)
        # Now we need to run the embeddings through the LSTM layer
        rnn_output, hidden = self.gru(embedded, hidden)
        # Compute attention and context vector
        context_vector, attention = self.attn(hidden, annotations.detach())
        #print("RNN PUT", rnn_output.shape, "CONTEXT", context_vector.shape)
        output = self.attn_hn(torch.cat([rnn_output.squeeze(), context_vector.squeeze()], dim=1))
        #print("OUTPUT", output.shape)
        # Next run tensor through a fully connected layer that maps the LSTM outputs to the predicted classes
        #output = self.fc(output)
        #print("OUTPUT", output.shape)
        # Finally map the outputs of the LSTM layer to a probability distribution
        output = self.softmax(output)
        # Return the prediction and the hidden state of the decoder
        return output, hidden, attention

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)
    
    
class AdditiveAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional, max_length, use_last_n_states=1):
        super(AdditiveAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.max_length = max_length
        self.last_n_annotations = (2 * self.hidden_size if self.bidirectional else self.hidden_size) * use_last_n_states
        self.last_n_hidden_states = (2 if self.bidirectional else 1) * use_last_n_states
        energy_input_dim = 2 * hidden_size * use_last_n_states
        if bidirectional:
            energy_input_dim *= 2
        self.energy = torch.nn.Linear(energy_input_dim, 1, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, hidden, annotations, logging=False):
        # Combine information from encoder annotations with current hidden state
        h_st = combine_encoder_annotations_and_hidden_state(
            hidden[-self.last_n_hidden_states:, :, :],
            annotations[:, :, -self.last_n_annotations:]
        )
        # This layer calculates an energy value e_ij for every word.
        # The energy is just a linear combination of the decoder hidden state
        # and the encoder annottations
        #energy = self.energy(h_st)
        energy = torch.bmm(annotations, concat_hidden_states(hidden).unsqueeze(dim=2))
        print("ENERGY SHAPE", energy.shape) if logging else None
        # Normalize energy values
        attention = self.softmax(energy)
        # batch matrix multiplication of attention and encoder outputs
        context = torch.bmm(attention.permute(0, 2, 1), annotations)
        print("CONTEXT VECTOR SHAPE", context.shape) if logging else None
        return context, attention        


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, max_length=20, num_layers=3, dropout=0.1, bidirectional=True, pos_encoding=False, device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pos_encoding = pos_encoding

        self.dropout = torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if self.bidirectional:
            self.attn = torch.nn.Linear(self.embedding_dim + self.hidden_size * 2, self.max_length)
        else:
            self.attn = torch.nn.Linear(self.embedding_dim + self.hidden_size, self.max_length)
        self.attn_combine = torch.nn.Linear(self.embedding_dim + self.hidden_size, self.hidden_size)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        if self.bidirectional:
            self.fc = torch.nn.Linear(2*hidden_size, vocab_size)
        else:
            self.fc = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, x, coordinates, position, hidden, encoder_outputs):
        # First run the input sequences through an embedding layer
        (hn, cn) = hidden
        embedded = self.embedding(x)
        # Add positional encoding to the embedding
        if self.pos_encoding:
            embedded = embedded + position
        embedded = self.dropout(embedded)
        _log("Shape of encoder outputs", encoder_outputs.shape)
        _log("Shape of embedding", embedded.shape)
        _log("Shape of hidden", hn.shape, cn.shape)
        _log("Shape of last hidden", hn[-1].shape)
        # Merge hidden states of LSTM layers
        hn_list = []
        hn_list.append(hn[-1])
        if self.bidirectional:
            hn_list.append(hn[-2])
        #for i in range(len(hn)):
        #    hn_list.append(hn[i])
        c = torch.cat([embedded.squeeze()] + hn_list, dim=1)
        _log("Cat shape", c.shape)
        attn_weights = self.attn(torch.cat([embedded.squeeze()] + hn_list, dim=1))
        _log("#1 Shape of attention weights", attn_weights.shape)
        attn_weights = F.softmax(attn_weights, dim=1)
        _log("#2 Shape of attention weights", attn_weights.shape)
        _log("#3 Shape of attention weights unsqueezed", attn_weights.unsqueeze(dim=1).shape)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(dim=1),
            encoder_outputs
        )
        _log("Attention applied shape", attn_applied.shape)
        output = torch.cat((embedded, attn_applied), dim=2)
        _log("Merge encoder outputs with attention appied", output.shape)
        output = self.attn_combine(output)
        _log("After attention combine", output.shape)
        output = F.leaky_relu(output)
        _log("Output was passed through relu activation function", output.shape, hn.shape, cn.shape)
        output, hidden = self.lstm(output, hidden)
        _log("Shape after LSTM", output.shape, hidden[0].shape, hidden[1].shape)
        output = F.log_softmax(self.fc(output), dim=2)
        _log("Shape after FC", output.shape)
        return output, hidden, attn_weights

    def initHidden(self, batch_size, device='cpu'):
        if self.bidirectional:
            num_layers = 2 * self.num_layers
        else:
            num_layers = self.num_layers
        return torch.zeros(num_layers, batch_size, self.hidden_size).to(device)
