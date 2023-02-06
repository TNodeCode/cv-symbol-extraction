import torch
from seqgen.model import transformer, embedding
from seqgen.vocabulary import *

# check on which device the model should run
if torch.cuda.device_count():
    device = "cuda"
else:
    device = "cpu"

num_layers = 2
embedding_dim = 128
batch_size = 2
max_length = 50
heads = 8
dropout = 0

vocab_in = Vocabulary(vocab_filename="seqgen/vocab_in.txt")
vocab_out = Vocabulary(vocab_filename="seqgen/vocab_out.txt")

# Transformer model
model = transformer.Transformer(
    encoder_embedding_type=embedding.EmbeddingType.COORDS_DIRECT,
    src_vocab_size=len(vocab_in),
    trg_vocab_size=len(vocab_out),
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    heads=heads,
    dropout=dropout,
    src_pad_idx=1e10,
    trg_pad_idx=1e10,
    device=device
).to(device)

checkpoint_file = "transformer_realdata2.pt"
checkpoint = torch.load(checkpoint_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("MODEL LOADED")


def predict(input_seqs, coordinates, target_seqs):
    with torch.no_grad():
        output = model(input_seqs, target_seqs, coordinates)
        # Get the predicted classes of the model
        topv, topi = output.topk(1, dim=2)
        return topi.squeeze()


def predict_sequentially(input_seqs, coordinates):
    prediction = torch.zeros(
        (input_seqs.size(0), input_seqs.size(1)-1)).to(torch.int64)
    for i in range(max_length-1):
        output = predict(input_seqs, coordinates, prediction).unsqueeze(dim=0)
        prediction[:, i] = output[:, i]
    return vocab_out.decode_sequence(prediction.squeeze().cpu().numpy())
