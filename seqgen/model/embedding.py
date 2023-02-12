import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sincos_position_encoding(max_length, embedding_dim, n=10000, device='cpu'):
    """
    Encode positions with sine and cosine functions
    """
    P = torch.zeros((max_length, embedding_dim)).to(device)
    for k in range(max_length):
        for i in torch.arange(int(embedding_dim/2)):
            denominator = n ** (2*i/embedding_dim)
            P[k, 2*i] = torch.sin(k/denominator)
            P[k, 2*i+1] = torch.cos(k/denominator)
    return P

    
class EmbeddingType:
    NONE="none"
    COORDS_DIRECT = "coords_direct"
    COORDS_RESIDUAL = "coords_residual"
    POS_TRIGENC = "pos_trigenc"
    POS_SUBSPACE = "pos_subspace"
    
    @staticmethod
    def embedding_layer(embedding_type):
        embedding_types = {
            "coords_direct": DirectCoordinateEmbedding,
            "coords_residual": ResidualCoordinateEmbedding,
            "pos_trigenc": PositionEncodingEmbedding,
            "pos_subspace": PositionSubspaceEmbedding,
        }
        return embedding_types[embedding_type]


class DirectCoordinateEmbedding(nn.Module):
    """
    Module that creates a concatination of a word embedding and the coordinates
    """
    def __init__(self, vocab_size, embedding_dim, dropout_prob=0.1, coordinates_dim=4, device='cpu', **kwargs):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(DirectCoordinateEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.coordinates_dim = coordinates_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim - coordinates_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x, coordinates):
        """
        Create an embedding that contains the embedded input word and the coordinates
        
        Keyword arguments:
        x           -- The indices of the tokens in the input sequence
        coordinates -- the coordinates of the tokens in the input sequence

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # First run the input sequences through an embedding layer
        embedded = self.dropout(self.embedding(x))
        # Concatenate embeddings with coordinates
        if (len(coordinates.shape) < len(embedded.shape)):
            coordinates = coordinates.unsqueeze(dim=1)
        return torch.cat([embedded, coordinates], dim=2)


class ResidualCoordinateEmbedding(nn.Module):
    """
    Module that creates a concatination of a word embedding and the coordinates
    """
    def __init__(self, vocab_size, embedding_dim, dropout_prob=0.1, coordinates_dim=4, device='cpu', **kwargs):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(ResidualCoordinateEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.coordinates_dim = coordinates_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim - coordinates_dim)
        self.pos_emb = nn.Linear(coordinates_dim, coordinates_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x, coordinates):
        """
        Create an embedding that contains the embedded input word and the coordinates
        
        Keyword arguments:
        x           -- The indices of the tokens in the input sequence
        coordinates -- the coordinates of the tokens in the input sequence

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # First run the input sequences through an embedding layer
        embedded = self.dropout(self.embedding(x))
        # Add residual to the coordinates
        coordinate_embedding = coordinates + F.tanh(self.pos_emb(coordinates))
        # Concatenate embeddings with coordinates
        return torch.cat([embedded, coordinate_embedding.unsqueeze(dim=1)], dim=2)


class PositionEncodingEmbedding(nn.Module):
    """
    Module that creates an embedding by adding the word embedding and the position encoding
    """
    def __init__(self, vocab_size, embedding_dim, max_length, dropout_prob=0.1, device='cpu', **kwargs):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        max_length    -- Maximum length of the sequence
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(PositionEncodingEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x, pos):
        """
        Create an embedding that contains the embedded input word and the coordinates
        
        Keyword arguments:
        x   -- The indices of the tokens in the input sequence
        pos -- the positions (integers from 1 to max_length) of the input tokens

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # Get the sine and cosine encodings of the positions
        pos_enc = get_sincos_position_encoding(
                self.max_length,
                self.embedding_dim,
                device=self.device
            )[pos, :]
        # Run the input token through the embedding layer and then add the position encoding to it
        return self.dropout(
            self.embedding(x) +
            pos_enc
        )


class PositionSubspaceEmbedding(nn.Module):
    """
    Module that creates a word embedding and a position embedding
    """
    def __init__(self, vocab_size, embedding_dim, max_length, dropout_prob=0.1, pos_embedding_dim=4, device='cpu', **kwargs):
        """    
        Keyword arguments:
        vocab_size    -- The number of unique tokens in the vocabulary
        embedding_dim -- dimension of the embedding that this function will output
        dropout_prob  -- probability of decativating neurons randomly
        device        -- Device that the operations should run on (cpu | cuda)
        """        
        super(PositionSubspaceEmbedding, self).__init__()
        
        # Hyperparameters
        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.dropout_prob = dropout_prob
        
        # Layers
        self.word_embedding = torch.nn.Embedding(vocab_size, embedding_dim - pos_embedding_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, pos_embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        
    def forward(self, x, pos):
        """
        Create an embedding that contains the embedded input word and the embedded position
        
        Keyword arguments:
        x           -- The index of the token in the sequence
        pos         -- the position of the token in the sequence

        Returns:
        This functions returns an embedding that contains the information aboout a token and its position        
        """
        # First run the input sequences through an embedding layer
        embedded_word = self.dropout(self.word_embedding(x))
        # Then run the positions through an embedding layer
        embedded_pos = self.dropout(self.pos_embedding(pos))
        # Concatenate word embeddings with position embeddings
        return torch.cat([embedded_word, embedded_pos], dim=2)
    
    
class PatchEmbed(nn.Module):
    """
    Image patch embedding module
    """
    def __init__(self, img_size, channels=1, patch_size=4, embed_dim=96, norm_layer=None, device='cpu', **kwargs):
        """
        Create image to patch embedding module.

        Parameters:
            - img_size (int): Image size.
            - channels (int) [1]: Number of input channels.
            - patch_size (int) [4]: Patch token size.
            - embed_dim (int) [96]: Size of embedding vectors.
            - norm_layer (nn.Module|None) [None]: Normalization layer.

        """
        super().__init__()
        ############################################################
        ###                  START OF YOUR CODE                  ###
        ############################################################

        # Hyperparameters
        self.img_size = img_size
        self.channels = channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        self.device = device
        
        # Layers
        self.embedding = nn.Linear(patch_size*patch_size, embed_dim)

        ############################################################
        ###                   END OF YOUR CODE                   ###
        ############################################################

    def forward(self, x):
        """
        Create patch embeddings from input images.

        Parameters:
            - x (torch.Tensor): (batch_size, [channels,] height, width).

        Returns:
            - x (torch.Tensor): (batch_size, num_patches, embed_dim).

        """
        # If there is no explicit dimension for the channel then add one
        if (len(x.shape) == 3):
            x = x.unsqueeze(dim=1)
            
        # Extrakt shape informtion
        batch_size, channels, height, width = x.shape
        
        # Split images into patches (patches will be of shape (batch_size,n_patches,channels,patch_size^2)
        # Each list items contains the same region of each image in the batch
        size = self.patch_size # patch size
        n_patches = (self.img_size // self.patch_size)**2 * self.channels
        patches = x.unfold(2, size, size).unfold(3, size, size).reshape(batch_size, n_patches, size*size)
        
        # Embed each patch (result will be of shape (batch_size,n_patches,channels,embed_dim))
        embeddings = self.embedding(patches)
        
        if self.norm_layer:
            embeddings = self.norm_layer(embeddings)
        return embeddings