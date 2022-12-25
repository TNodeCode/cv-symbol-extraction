import torch
import random as r
import json
from seqgen.vocabulary import *
from seqgen.preprocess import *

digits_in = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
digits_out = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

letters_in = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
              "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
letters_out = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
               "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

operators_in = ["op_plus", "op_minus", "op_multiply", "op_divide"]
operators_out = ["+", "-", "\cdot", "/"]

tmpl_in = ["sum", "LETTER", "=", "DIGIT", "LETTER"]
tmpl_out = ["\\sum", "\\limits", "_",
            "{", "LETTER", "=", "DIGIT", "}", "^", "LETTER"]

def gen_default_box(x0, y0, box_height, box_width, seq_in, seq_out, idx, in_voc, out_voc):
    _box_height = r.uniform(0.9, 1.1) * box_height
    _box_width = r.uniform(0.9, 1.1) * box_height
    _x0 = r.uniform(0.9, 1.1) * x0
    _y0 = r.uniform(0.9, 1.1) * y0
    # Append tokens to input and output sequence
    seq_in.append([in_voc[idx], _x0, _y0, _x0 + _box_height, _y0 + _box_width])
    seq_out.append(out_voc[idx])
    return seq_in, seq_out, _box_width

def gen_exponent_box(x0, y0, box_height, box_width, seq_in, seq_out, idx, in_voc, out_voc):
    _box_height = r.uniform(0.4, 0.6) * box_height
    _box_width = r.uniform(0.9, 1.1) * box_height
    _x0 = r.uniform(0.7, 1.1) * x0
    _y0 = r.uniform(0.0, 0.8) * y0
    # Append tokens to input and output sequence
    seq_in.append([in_voc[idx], _x0, _y0, _x0 + _box_height, _y0 + _box_width])
    seq_out.append('^')
    seq_out.append(out_voc[idx])
    return seq_in, seq_out, _box_width

def gen_index_box(x0, y0, box_height, box_width, seq_in, seq_out, idx, in_voc, out_voc):
    _box_height = r.uniform(0.4, 0.6) * box_height
    _box_width = r.uniform(0.9, 1.1) * box_height
    _x0 = r.uniform(0.7, 1.1) * x0
    _y0 = r.uniform(1.3, 2.0) * y0
    # Append tokens to input and output sequence
    seq_in.append([in_voc[idx], _x0, _y0, _x0 + _box_height, _y0 + _box_width])
    seq_out.append('_')
    seq_out.append(out_voc[idx])
    return seq_in, seq_out, _box_width

def generate_random_sequence(in_voc: list, out_voc: list, continue_prob=0.9, max_length=20, padding=True, swap_prob=0.5, swap_times=5):
    """
    Generate a random training sample for a sequence-to-sequence model

    Parameters:
    in_voc: List of tokens that can be used for the input sequence
    out_voc: List of tokens that can be used for the target sequence
    continue_prob: Probability that one more token will be appended to the sample sequence
    max_length: Maximum length of the sequence
    """
    seq_in = [['<start>', 0, 0, 0, 0]]
    seq_out = ['<start>']

    # This is the position of the upper left corner of the current token
    x0, y0 = (100, 100)
    # default width and height of a bounding box
    box_width, box_height = 30, 60

    while True:
        # Select a random token from the vocabulary
        random_index = r.randint(0, len(in_voc) - 1)
        random_choice = r.random()
        
        # Generate random boxes    
        if random_choice < 0.8:
            gen = gen_default_box
        elif random_choice < 0.9 and len(seq_out) < max_length - 2:
            gen = gen_index_box
        elif len(seq_out) < max_length - 2:
            gen = gen_exponent_box
        else:
            gen = gen_default_box
        
        seq_in, seq_out, width = gen(x0, y0, box_height, box_width, seq_in, seq_out, random_index, in_voc, out_voc)
        x0 = x0 + 1.1 * width

        # Check if the sequence should be continued
        if r.random() > continue_prob or len(seq_in) >= (max_length - 1) or len(seq_out) >= (max_length - 1):
            break

    # Pad sequence with zeros
    if padding:
        for i in range(max_length - len(seq_in)):
            seq_in.append(['<end>', 0, 0, 0, 0])
        for i in range(max_length - len(seq_out)):
            seq_out.append('<end>')
            
    # Randomly swap positions
    for i in range(swap_times):
        if r.random() < swap_prob:
            seq_in = random_swap(seq_in)

    return seq_in, seq_out


def gen_seq(tmpl_in, tmpl_out):
    lst_in, lst_out = [], []
    for tmpl, lst in [(tmpl_in, lst_in), (tmpl_out, lst_out)]:
        for t in tmpl:
            if t == "LETTER":
                lst.append(r.choice(letters))
            elif t == "DIGIT":
                lst.append(r.choice(digits))
            elif t == "OPERATOR":
                lst.append(r.choice(operators))
            else:
                lst.append(t)
    return lst_in, lst_out


def save_as_json(samples, filename="samples.json"):
    """
    Save samples as JSON file
    """
    with open(filename, "w") as fp:
        json.dump(samples, fp, indent=2)
        
        
def random_swap(lst, i=None):
    """
    Swap the positions of two elements randomly.
    If you pass a list like ['a','b','c'] to this function you may get ['b','a','c'] back
    """
    if i is None:
        i = r.randint(1, len(lst) - 2)
    j = i + 1
    
    # Tensors and lists have to treated differently, because tensors are implemented in C
    # and use pointers, which leads to a different behaviour than lists
    if type(lst) == list:
        lst[i], lst[j] = lst[j], lst[i]
    elif type(lst) == torch.Tensor:
        lst = torch.clone(lst)
        vi = torch.clone(lst[i])
        vj = torch.clone(lst[j])
        lst[i] = vj
        lst[j] = vi
    return lst

def generator(num_samples=5, max_length=10, continue_prob=0.95, swap_prob=0.5, swap_times=5):
    """
    Generate synthetic training samples for a sequence-to-sequence model.
    A training sample consists of an input sequence (feature sequence) and the desired output sequence (target sequence).
    The input sequence items consist of an index from the input vocabluary and four coordinates (x0,y0,x1,y1).
    The output sequence items only contain the class index of the output vocabluary.

    Parameters:
    num_samples: Number of training samples that will be created
    """
    inputs, outputs = [], []
    for i in range(num_samples):
        feature_seq, target_seq = generate_random_sequence(
            in_voc=digits_in+operators_in,
            out_voc=digits_out+operators_out,
            continue_prob=continue_prob,
            max_length=max_length,
            swap_prob=swap_prob,
            swap_times=swap_times
        )
        inputs.append(feature_seq)
        outputs.append(target_seq)
    return inputs, outputs


def generate_synthetic_training_data(num_samples=16, max_length=10, continue_prob=0.95, swap_prob=0.5, swap_times=5, device='cpu'):
    vocab_in = Vocabulary(vocab_filename="seqgen/vocab_in.txt")
    vocab_out = Vocabulary(vocab_filename="seqgen/vocab_out.txt")
    features, targets = generator(num_samples, max_length=max_length, continue_prob=continue_prob, swap_prob=swap_prob, swap_times=swap_times)
    features = encode_classes_of_bboxes(features, vocab_in)
    features = normalize_coordinates(features)
    targets = encode_latex_tokens(targets, vocab_out)
    return torch.tensor(features).to(device), torch.tensor(targets).to(device)
