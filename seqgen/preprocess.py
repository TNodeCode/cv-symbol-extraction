import torch


def min_max(x, min_val, max_val):
    """
    Min-Max Scaling
    """
    return (x - min_val) / (max_val - min_val)


def encode_classes_of_bboxes(sequences, vocab):
    """
    Encode an input sequence: Replace classes of characters with indices
    :param sequences: List of list of characters
    :param vocab:     Vocabulary object that can translate charactters to indices and vice versa
    """
    encoded_sequences = []
    # Iterate over batch of sequences
    for seq in sequences:
        encoded_seq = []
        # Iterate over bounding boxes of sequence
        for cls, x0, y0, x1, y1 in seq:
            # Encode classes of bounding boxes
            encoded_seq.append([vocab(cls), x0, y0, x1, y1])
        encoded_sequences.append(encoded_seq)
    return encoded_sequences


def encode_latex_tokens(sequences, vocab):
    encoded_sequences = []
    # Iterate over batch of sequences
    for seq in sequences:
        encoded_seq = []
        # Iterate over tokens of sequence
        for cls in seq:
            # Encode classes of tokens
            encoded_seq.append(vocab(cls))
        encoded_sequences.append(encoded_seq)
    return encoded_sequences


def normalize_coordinates(feature_seqs):
    """
    Normalize coordinates
    """
    encoded_seqs = []
    for feature_seq in feature_seqs:
        encoded_seq = []
        min_x, min_y, max_x, max_y = 1e9, 1e9, -1e9, -1e9
        for cls, x0, y0, x1, y1 in feature_seq:
            min_x = min(min_x, x0)
            max_x = max(max_x, x1)
            min_y = min(min_y, y0)
            max_y = max(max_y, y1)
        for i, (cls, x0, y0, x1, y1) in enumerate(feature_seq):
            encoded_seq.append([cls, min_max(x0, min_x, max_x), min_max(
                y0, min_y, max_y), min_max(x1, min_x, max_x), min_max(y1, min_y, max_y)])
        encoded_seqs.append(encoded_seq)
    return encoded_seqs


def get_coordinate_encoding(coordinates, d, max_length, n=10000, device='cpu'):
    batch_size = coordinates.shape[0]
    seq_len = coordinates.shape[1]
    P = torch.zeros((batch_size, seq_len, d)).to(device)
    for k in range(seq_len):
        for i in torch.arange(int(d/8)):
            denominator = n ** (2*i/d)
            P[:, k, 8*i+0] = torch.sin(coordinates[:, k, 0]*max_length/denominator)
            P[:, k, 8*i+1] = torch.cos(coordinates[:, k, 0]*max_length/denominator)
            P[:, k, 8*i+2] = torch.sin(coordinates[:, k, 1]*max_length/denominator)
            P[:, k, 8*i+3] = torch.cos(coordinates[:, k, 1]*max_length/denominator)
            P[:, k, 8*i+4] = torch.sin(coordinates[:, k, 2]*max_length/denominator)
            P[:, k, 8*i+5] = torch.cos(coordinates[:, k, 2]*max_length/denominator)
            P[:, k, 8*i+6] = torch.sin(coordinates[:, k, 3]*max_length/denominator)
            P[:, k, 8*i+7] = torch.cos(coordinates[:, k, 3]*max_length/denominator)
    return P
