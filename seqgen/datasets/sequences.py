import torch
from torch.utils.data import Dataset
import seqgen.seq_gen as g


class SyntheticSequenceDataset(Dataset):
    def __init__(self, vocab_in, vocab_out, max_length, batch_size, continue_prob=0.99, additional_eos=False, device="cpu") -> None:
        self.max_length = max_length
        self.additional_eos = additional_eos
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        self.batch_size = batch_size
        self.continue_prob = continue_prob
        self.device = device
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __getitem__(self, index: int):
        features, target_seqs = g.generate_synthetic_training_data(
            num_samples=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            continue_prob=self.continue_prob,
            swap_times=self.max_length
        )
        input_seqs = torch.tensor(features[:, :, 0]).to(torch.int64)
        coordinates = torch.tensor(features[:, :, 1:])
        
        if self.additional_eos:
            input_seqs = torch.cat([
                input_seqs,
                (torch.ones(input_seqs.size(0), 1)*2).to(self.device)
            ], dim=1).to(torch.int64)
            target_seqs = torch.cat([
                target_seqs,
                (torch.ones(target_seqs.size(0), 1)*2).to(self.device)
            ], dim=1).to(torch.int64)
            coordinates = torch.cat([
                coordinates,
                torch.zeros(coordinates.size(0), 1, coordinates.size(2)).to(self.device)
            ], dim=1)
        
        return input_seqs, coordinates, target_seqs