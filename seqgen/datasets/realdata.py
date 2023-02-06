import torch
from torch.utils.data import Dataset
import numpy as np
from seqgen.parser import *
from seqgen.preprocess import *
from seqgen.seq_gen import *
import seqgen.symbol_replacement as symbol_replacement



class RealSequencesDataset(Dataset):
    def __init__(self, filename, vocab_in, vocab_out, max_length, batch_size, sos_idx=0, eos_idx=1, additional_eos=False, device="cpu") -> None:
        self.max_length = max_length
        self.additional_eos = additional_eos
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        self.eos_idx = eos_idx
        self.batch_size = batch_size
        self.device = device
        # read the label file
        content = read_label_file(filename)
        # get the formulas, classes and coordinates
        formulas, boxes = parse_label_file(content)
        self.boxes = boxes
        # get the keys
        keys = np.array(list(sorted(list(vocab_out.word2idx.keys()), key=lambda k: len(k))))[::-1]
        # get the key lengths
        lens = np.array(list(map(lambda k: len(k), keys)))
        self.input_seqs = []
        self.coordinates = []
        self.target_seqs = []
        formula_index = 0
        # parse the formulas
        for i, f in enumerate(formulas):
            # parse latex string
            parsed_formula = parse_formula(f, keys)
            # check if parsed formula contains more tokens than the given maximum length
            if len(parsed_formula) <= self.max_length and self.boxes[i].shape[0] <= self.max_length:
                # encode the latex tokens
                target_seq = np.array(self.vocab_out.encode_sequence(parsed_formula))
                # pad the latex list with zeros at the right side
                target_seq = np.pad(target_seq, (0, self.max_length - len(target_seq)), mode='constant')
                # replace with zeros with '<eos>' index
                target_seq[target_seq == 0] = self.eos_idx
                # append encoded latex tokens to the list of target sequences
                self.target_seqs.append(target_seq)
                # convert input sequence to integers (because they are float at the moment)
                input_seq = self.boxes[i][:, 0].astype(np.int32)
                # extract coordinates rom XcYcWH to XYXY format
                coords = self.reformat_coords(self.boxes[i][:, 1:5])
                # sort the input sequences and the coordinates by the x0 values
                idx_coords_sorted = np.argsort(coords[:, 0])
                coords = coords[idx_coords_sorted, :]
                input_seq = input_seq[idx_coords_sorted]
                # remove the formula boxes
                mask = input_seq != formula_index
                input_seq = input_seq[mask]
                coords = coords[mask, :]
                # Check that each box has coordinates, otherwise raise an error
                assert len(input_seq) == len(coords), "There should be one box for each token of the input sequence"
                # normalize the coordinates (apply minmax to them)
                coords = np.array(normalize_coordinates(np.array([coords]), contains_class=False)).squeeze()
                # Check if coords are still a 2 array (if sequence contains only 1 token that will not be the case)
                if (len(coords.shape) == 1):
                    coords = coords.reshape(-1, 4)
                # also pad the end of the coordinates list
                coords = self.pad_coordinates(coords, max_length)
                # now append the coordinates of the current sequence to the list of coordinates
                self.coordinates.append(coords)
                # Add 3 to the input sequences, because the YOLO algorithm doesn't add classes for '<sos>', '<eos>' and '<unk>'
                input_seq += 3
                # pad the input sequence
                input_seq = np.pad(input_seq, (0, self.max_length - len(input_seq)), mode='constant')
                # replace the zeros from the padding step with the encoded '<eos>' token
                input_seq[input_seq == 0] = self.eos_idx
                # add the encoded input sequence to the dataset
                self.input_seqs.append(input_seq)
        # transform lists of sequences to ndarray
        self.input_seqs = np.concatenate([
            np.ones((len(self.input_seqs),1))*sos_idx,
            np.array(self.input_seqs),
            np.ones((len(self.input_seqs),1))*eos_idx,
        ], axis=1)
        self.coordinates = np.concatenate([
            np.zeros((len(self.input_seqs),1,4)),
            np.array(self.coordinates),
            np.zeros((len(self.input_seqs),1,4)),
        ], axis=1)
        self.target_seqs = np.concatenate([
            np.ones((len(self.input_seqs),1))*sos_idx,
            np.array(self.target_seqs),
            np.ones((len(self.input_seqs),1))*eos_idx,
        ], axis=1)
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __getitem__(self, index: int):
        idx = list(np.random.randint(0, self.input_seqs.shape[0], size=(self.batch_size,)))
        _coords = add_noise_to_coordinates(self.coordinates[idx])
        input_seqs, coordinates, target_seqs = torch.tensor(self.input_seqs[idx]).to(torch.int64), torch.tensor(self.coordinates[idx]).to(torch.float32), torch.tensor(self.target_seqs[idx]).to(torch.int64)
        #input_seqs, target_seqs = symbol_replacement.generate_new_sequences(input_seqs, target_seqs, self.vocab_in, self.vocab_out)
        
        return input_seqs.to(self.device), coordinates.to(self.device), target_seqs.to(self.device)
    
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
        """
        Pad coordinates with zeros
        """
        length = coordinates.shape[0]
        pad_size = max_length - length
        return np.concatenate((coordinates, np.zeros((pad_size, 4))), axis=0)