from vocabulary import Vocabulary
import torch
import seqgen.seq_gen as g
import random
import matplotlib.pyplot as plt
import seaborn as sns
from seqgen.model import rnn
from seqgen.vocabulary import *
from seqgen.model import transformer, embedding
from seqgen.datasets.sequences import *
from seqgen.datasets.realdata import RealSequencesDataset

torch.autograd.set_detect_anomaly(True)

if torch.cuda.device_count():
    device="cuda"
else:
    device="cpu"
print("Device", device)

use_real_dataset=True
lr=1e-3
num_layers=2
embedding_dim=128
batch_size=2
max_length=50
heads=8
dropout=0

vocab_in = Vocabulary(vocab_filename="vocab_in.txt", encoded_txt_filename="encoded.txt")
vocab_out = Vocabulary(vocab_filename="vocab_out.txt", encoded_txt_filename="encoded.txt")

if use_real_dataset:
    dataset = RealSequencesDataset(filename="label.txt", vocab_in=vocab_out, vocab_out=vocab_out, max_length=max_length-1, batch_size=batch_size, device=device)
else:
    dataset = SyntheticSequenceDataset(vocab_in, vocab_out, max_length, batch_size, continue_prob=0.95, additional_eos=True, device=device)

input_seqs, coordinates, target_seqs = dataset[0]
print(input_seqs, coordinates, target_seqs.shape)

print(input_seqs[0, :-1])
print(target_seqs[0, :-1])
print(target_seqs[0, 1:])

load_from_checkpoint = True
checkpoint_file = "transformer_2023-02.pt2"

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

# Initialize optimizer for encoder and decoder
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# Loss function
criterion = torch.nn.NLLLoss()

# Load model weights from checkpoint
if load_from_checkpoint:
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

