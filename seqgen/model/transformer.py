import torch
import torch.nn as nn
import torch.nn.functional as F
from seqgen.preprocess import *
from seqgen.model.classifier import *


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
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
class CoordinateEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.device = device

    def forward(self, coordinates):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return get_coordinate_encoding(
            coordinates,
            d=self.d_model,
            max_length=self.max_length,
            device=self.device
        )
        
        
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
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
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
        
        
class EncoderCoordsEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_length, coord_embedding_dim=0):
        super(EncoderCoordsEmbedding, self).__init__()
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.coord_embedding_dim = coord_embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout_prob = dropout,
        self.max_length = max_length
        
        # Layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim-coord_embedding_dim)
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
        out = self.dropout(torch.cat([self.word_embedding(x), coordinates], dim=2))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
        
        
class EncoderTrigPosEnc(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(EncoderTrigPosEnc, self).__init__()
        
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.heads = heads
        self.device = device
        self.forward_expansion = forward_expansion
        self.dropout_prob = dropout,
        self.max_length = max_length
        
        # Layers
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, coordinates):
        N, seq_length = x.shape
        out = self.dropout(
            self.word_embedding(x) +
            get_coordinate_encoding(coordinates, d=self.embedding_dim, max_length=self.max_length).to(self.device)
        )
        
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
        self.cls = Classifier(
            trg_vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            softmax_dim=2
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            
        out = self.cls(x)
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
        
        self.pos_emb = nn.Linear(6, 6)
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            coord_embedding_dim=6,
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
        _coords = torch.zeros(coordinates.size(0), coordinates.size(1), 6)
        _coords[:, :, 0:4] = coordinates
        _coords[:, :, 4:5] = (coordinates[:, :, 2] - coordinates[:, :, 0]).unsqueeze(dim=2) # x1 - x0
        _coords[:, :, 5:6] = (coordinates[:, :, 3] - coordinates[:, :, 1]).unsqueeze(dim=2) # y1 - y0
        enc_src = self.encoder(src, src_mask, _coords + self.pos_emb(_coords))
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
                
        
class TransformerCoordsEmbedding(nn.Module):
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
        super(TransformerCoordsEmbedding, self).__init__()
        self.device = device
        
        self.pos_emb = nn.Linear(6, 6)
        self.encoder = EncoderCoordsEmbedding(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            coord_embedding_dim=6,
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
        _coords = torch.zeros(coordinates.size(0), coordinates.size(1), 6)
        _coords[:, :, 0:4] = coordinates
        _coords[:, :, 4:5] = (coordinates[:, :, 2] - coordinates[:, :, 0]).unsqueeze(dim=2) # x1 - x0
        _coords[:, :, 5:6] = (coordinates[:, :, 3] - coordinates[:, :, 1]).unsqueeze(dim=2) # y1 - y0
        enc_src = self.encoder(src, src_mask, _coords + F.tanh(self.pos_emb(_coords)))
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out    
                
        
class EncoderModel(nn.Module):
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
        super(EncoderModel, self).__init__()
        self.device = device
        
        self.pos_emb = nn.Linear(6, 6)
        self.encoder = EncoderCoordsEmbedding(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            coord_embedding_dim=6,
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
        self.fc_out = nn.Linear(embedding_dim, trg_vocab_size)
        self.softmax = nn.Softmax(dim=2)

        
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
        _coords = torch.zeros(coordinates.size(0), coordinates.size(1), 6)
        _coords[:, :, 0:4] = coordinates
        _coords[:, :, 4:5] = (coordinates[:, :, 2] - coordinates[:, :, 0]).unsqueeze(dim=2) # x1 - x0
        _coords[:, :, 5:6] = (coordinates[:, :, 3] - coordinates[:, :, 1]).unsqueeze(dim=2) # y1 - y0
        enc_src = self.encoder(src, src_mask, _coords + F.tanh(self.pos_emb(_coords)))
        out = self.softmax(self.fc_out(enc_src))
        return out                

    
class TransformerCoordsEmbeddingDirect(nn.Module):
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
        super(TransformerCoordsEmbeddingDirect, self).__init__()
        self.device = device
        
        self.encoder = EncoderCoordsEmbedding(
            vocab_size=src_vocab_size,
            embedding_dim=embedding_dim,
            coord_embedding_dim=6,
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
        _coords = torch.zeros(coordinates.size(0), coordinates.size(1), 6)
        _coords[:, :, 0:4] = coordinates
        _coords[:, :, 4:5] = (coordinates[:, :, 2] - coordinates[:, :, 0]).unsqueeze(dim=2) # x1 - x0
        _coords[:, :, 5:6] = (coordinates[:, :, 3] - coordinates[:, :, 1]).unsqueeze(dim=2) # y1 - y0
        _coords[:, :, [0,2]] = _coords[:, :, [0,2]] * 50
        enc_src = self.encoder(src, src_mask, _coords)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
                
        
class TransformerTrigPosEnc(nn.Module):
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
        super(TransformerTrigPosEnc, self).__init__()
        self.device = device
        
        self.encoder = EncoderTrigPosEnc(
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
        x_min = torch.min(coordinates[:, :, 0])
        x_max = torch.max(coordinates[:, :, 2])
        coordinates[:, :, [0,2]] = coordinates[:, :, [0,2]] * 50
        enc_src = self.encoder(src, src_mask, coordinates)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
        