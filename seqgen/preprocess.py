def min_max(x, min_val, max_val):
    """
    Min-Max Scaling
    """
    return (x - min_val) / (max_val - min_val)


def encode_classes_of_sequences(input_sequences, vocab_in, vocab_out):
    encoded_sequences = []
    for seq in input_sequences:
        feature_seq = []
        target_seq = []
        for cls, x0, y0, x1, y1 in seq["feature_seq"]:
            feature_seq.append([vocab_in(cls), x0, y0, x1, y1])
        for cls in seq["target_seq"]:
            target_seq.append(vocab_out(cls))
        encoded_sequences.append(
            {"feature_seq": feature_seq, "target_seq": target_seq})
    return encoded_sequences


def normalize_coordinates(feature_seq):
    """
    Normalize coordinates
    """
    min_x, min_y, max_x, max_y = 1e9, 1e9, -1e9, -1e9
    for cls, x0, y0, x1, y1 in feature_seq:
        min_x = min(min_x, x0)
        max_x = max(max_x, x1)
        min_y = min(min_y, y0)
        max_y = max(max_y, y1)
    for i, (cls, x0, y0, x1, y1) in enumerate(feature_seq):
        feature_seq[i] = [cls, min_max(x0, min_x, max_x), min_max(
            y0, min_y, max_y), min_max(x1, min_x, max_x), min_max(y1, min_y, max_y)]
    return feature_seq
