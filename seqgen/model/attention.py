import torch
import torch.nn as nn
import torch.nn.functional as F


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

    
class AttentionType:
    DOT = "dot"
    GENERAL = "general"
    CONCAT = "concat"
    
    @staticmethod
    def attention_layer(attention_type):
        attention_types = {
            "dot": DotAttention,
            "general": GeneralAttention,
            "concat": ConcatAttention,
        }
        return attention_types[attention_type]
    
    
class DotAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional, max_length, batch_size, use_last_n_states=1):
        super(DotAttention, self).__init__()
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi_factor = 2 if bidirectional else 1
        self.max_length = max_length
        energy_input_dim = (self.bi_factor*2) * hidden_size
        
        # Layers
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, hidden, annotations, logging=False):
        # Combine information from encoder annotations with current hidden state
        h_st = combine_encoder_annotations_and_hidden_state(
            hidden[-self.bi_factor:, :, :],
            annotations[:, :, :]
        )
        # This layer calculates an energy value e_ij for every word.
        # The energy is just a linear combination of the decoder hidden state
        # and the encoder annotations
        energy = torch.bmm(
            annotations,
            concat_hidden_states(hidden[-self.bi_factor:]).unsqueeze(dim=2)
        )
        # Normalize energy values
        attention = self.softmax(energy)
        # batch matrix multiplication of attention and encoder outputs
        context = torch.bmm(attention.permute(0, 2, 1), annotations)
        return context, attention
    
    
class GeneralAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional, max_length, batch_size, use_last_n_states=1):
        super(GeneralAttention, self).__init__()
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi_factor = 2 if bidirectional else 1
        self.max_length = max_length
        energy_input_dim = (self.bi_factor*2) * hidden_size
        
        # Layers
        self.Wa = torch.nn.Linear(self.bi_factor*hidden_size, self.bi_factor*hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, hidden, annotations, logging=False):
        # This layer calculates an energy value e_ij for every word.
        # The energy is just a linear combination of the decoder hidden state
        # and the encoder annotations
        energy = torch.bmm(
            annotations,
            self.Wa(concat_hidden_states(hidden[-self.bi_factor:])).unsqueeze(dim=2)
        )
        # Normalize energy values
        attention = self.softmax(energy)
        # batch matrix multiplication of attention and encoder outputs
        context = torch.bmm(attention.permute(0, 2, 1), annotations)
        return context, attention
    
    
class ConcatAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, bidirectional, max_length, batch_size, use_last_n_states=1):
        super(ConcatAttention, self).__init__()
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bi_factor = 2 if bidirectional else 1
        self.max_length = max_length
        energy_input_dim = (self.bi_factor*2) * hidden_size
        
        # Layers
        self.Wa = torch.nn.Linear(energy_input_dim, energy_input_dim)
        self.va = torch.nn.Parameter(torch.randn(batch_size, 2*self.bi_factor* hidden_size))
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, hidden, annotations, logging=False):
        # Combine information from encoder annotations with current hidden state
        h_st = combine_encoder_annotations_and_hidden_state(
            hidden[-self.bi_factor:, :, :],
            annotations[:, :, :]
        )
        energy = torch.bmm(F.tanh(self.Wa(h_st)), self.va.unsqueeze(2))
        # Normalize energy values
        attention = self.softmax(energy)
        # batch matrix multiplication of attention and encoder outputs
        context = torch.bmm(attention.permute(0, 2, 1), annotations)
        return context, attention