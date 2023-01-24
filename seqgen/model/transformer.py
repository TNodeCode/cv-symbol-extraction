import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super(SelfAttention, self).__init__()
        
        # Hyperparameters
        self.heads = heads
        self.head_dim = embedding_dim // heads
        self.embedding_dim = embedding_dim
        
        assert self.head_dim * heads == embedding_dim, "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embedding_dim)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Send values, keys and queries through linear layers (matrix multiplication)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Energy describes how important each input is for each output
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Normalize energy -> sum of all weights is one
        attention = torch.softmax(energy / (self.embedding_dim ** 0.5), dim=3)
        
        # Matrix product between attention and values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        
        out = self.fc_out(out)
        return out
        
        
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion*embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion*embedding_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
        
        
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout_prob = dropout,
        self.max_length = max_length
        
        # Layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, coordinates):
        N, seq_length = x.shape
        #positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        #out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        out = self.dropout(self.word_embedding(x) + coordinates)
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim, heads)
        self.norm = nn.LayerNorm(embedding_dim)
        self.transformer_block = TransformerBlock(embedding_dim, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout_prob = dropout,
        self.max_length = max_length
        
        # Layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embedding_dim, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        out = self.fc_out(x)
        return out
        
        
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_dim=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100
    ):
        super(Transformer, self).__init__()
        self.device = device
        
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )
        self.decoder = Decoder(
            vocab_size=trg_vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # (N, 1, 1, src_length)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg, coordinates):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, coordinates)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
        