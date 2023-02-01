import torch
from torch.utils.data import Dataset
import numpy as np
from seqgen.parser import *
from seqgen.preprocess import *


class RealSequencesDataset(Dataset):
    def __init__(self, vocab_in, vocab_out, max_length, batch_size, eos_idx=1, additional_eos=False, device="cpu") -> None:
        self.max_length = max_length
        self.additional_eos = additional_eos
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        self.eos_idx = eos_idx
        self.batch_size = batch_size
        self.device = device
        
        # read the label file
        content = read_label_file("data/val/label.txt")
        # get the formulas, classes and coordinates
        formulas, boxes = parse_label_file(content)
        self.boxes = boxes
        # get the keys
        keys = np.array(list(sorted(list(vocab_out.word2idx.keys()), key=lambda k: len(k))))[::-1]
        # get the key lengths
        lens = np.array(list(map(lambda k: len(k), keys)))
        # parse the formulas
        self.input_seqs = []
        self.coordinates = []
        self.target_seqs = []
        for i, f in enumerate(formulas):
            parsed_formula = parse_formula(f, keys)
            if len(parsed_formula) <= self.max_length:
                target_seq = np.array(self.vocab_out.encode_sequence(parsed_formula))
                target_seq = np.pad(target_seq, (0, self.max_length - len(target_seq)), mode='constant')
                target_seq[target_seq == 0] = self.eos_idx
                self.target_seqs.append(target_seq)
                input_seq = self.boxes[i][:, 0].astype(np.int32)
                coords = self.reformat_coords(self.boxes[i][:, 1:5])
                if (input_seq[-1] == 0):
                    input_seq = input_seq[:-1]
                    coords = coords[:-1]
                    assert len(input_seq) == len(coords), "There should be one box for each token of the input sequence"
                    #print("BEFORE", coords.shape)
                    coords = np.array(normalize_coordinates(np.array([coords]), contains_class=False)).squeeze()
                    # Check if coords are still a 2 array (if sequence contains only 1 token that will not be the case)
                    if (len(coords.shape) == 1):
                        coords = coords.reshape(-1, 4)
                    #print("AFTER", coords.shape, len(coords.shape))
                    coords = self.pad_coordinates(coords, max_length)
                    self.coordinates.append(coords)
                input_seq = np.pad(input_seq, (0, self.max_length - len(input_seq)), mode='constant')
                input_seq[input_seq == 0] = self.eos_idx
                self.input_seqs.append(input_seq)
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __getitem__(self, index: int):
        return self.input_seqs, self.coordinates, self.target_seqs
    
    def reformat_coords(self, coordinates):
        """
        Reformat coordinates from format (x_center, y_center, width, height) to (x0,y0,x1,y1)
        """
        coords = np.zeros_like(coordinates)
        coords[:, 0] = coordinates[:, 0] - 0.5 * coordinates[:, 2]
        coords[:, 2] = coordinates[:, 0] + 0.5 * coordinates[:, 2]
        coords[:, 1] = coordinates[:, 1] - 0.5 * coordinates[:, 3]
        coords[:, 3] = coordinates[:, 1] + 0.5 * coordinates[:, 3]
        return coords
    
    def pad_coordinates(self, coordinates, max_length):
        length = coordinates.shape[0]
        pad_size = max_length - length
        return np.concatenate((coordinates, np.zeros((pad_size, 4))), axis=0)