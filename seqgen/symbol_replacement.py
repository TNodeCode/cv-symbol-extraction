import string
import random
import numpy as np
import torch


digits_in = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
digits_out = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

letters_in = np.array(list(string.ascii_uppercase + string.ascii_lowercase))
letters_out = np.array(list(string.ascii_uppercase + string.ascii_lowercase))

greek_in = np.array(["\\alpha", "\\beta", "\\gamma", "\\Delta", "\mu", "\\theta", "\\pi", "\\lambda", "\\sigma", "\\phi"])
greek_out = np.array(["\\alpha", "\\beta", "\\gamma", "\\Delta", "\mu", "\\theta", "\\pi", "\\lambda", "\\sigma", "\\phi"])

operators_in = np.array(["+", "-", "/", "\\times", "=", "\\leq", "\\geq", "\\gt", "\\pm", "\\forall", "\\in", "\\lt", "\\exists", "\\neq"])
operators_out = np.array(["+", "-", "/", "\\times", "=", "\\leq", "\\geq", "\\gt", "\\pm", "\\forall", "\\in", "\\lt", "\\exists", "\\neq"])

functions_in = np.array(["\\sin", "\\cos", "\\sqrt", "\\log"])
functions_out = np.array(["\\sin", "\\cos", "\\sqrt", "\\log"])

symbols_with_limits_in = np.array(["\\sum", "\\int", "\\lim"])
symbols_with_limits_out = np.array(["\\sum", "\\int", "\\lim"])

dots_in = ["\\ldots"]

equivalents = [
    (letters_in, letters_out),
    (digits_in, digits_out),
    (operators_in, operators_out),
    (functions_in, functions_out),
    (greek_in, greek_out),
    (symbols_with_limits_in, symbols_with_limits_out),
]


def replace_symbols(seq_in, seq_out, vocab_in, vocab_out):
    # First make copies of the sequences
    seq_in = np.copy(seq_in)
    seq_out = np.copy(seq_out)
    # Iterate over equivalent classes
    for (class_in, class_out) in equivalents:
        # Encode the sequences if a vocabulary object was passed to the function
        if vocab_in:
            class_in = np.array(vocab_in.encode_sequence(class_in))
        if vocab_out:
            class_out = np.array(vocab_out.encode_sequence(class_out))
        # Find all symbols that belong to the given class in the input sequence and the output sequence
        mask_in = np.isin(seq_in, class_in)
        mask_out = np.isin(seq_out, class_out)
        # Find unique symbols that should be replaced
        symbols_unique = np.unique(seq_in[mask_in])
        # Iterate over unique symbols of the given class found in the input sequence
        for symbol in symbols_unique:
            # Get the index of the symbol in the list of symbols of the given class
            symbol_class_index = int(np.where(class_in == symbol)[0])
            # Get the corresponding symbol of the input and output sequence
            symbol_in = class_in[symbol_class_index]
            symbol_out = class_out[symbol_class_index]
            # Select a random symbol of the class the symbols should be replaced with
            new_class_index = random.randint(0, len(class_in)-1)
            # Replace symbols in the input sequence and output sequence
            seq_in[seq_in == symbol_in] = class_in[new_class_index]
            seq_out[seq_out == symbol_out] = class_out[new_class_index]
    return seq_in, seq_out


def generate_new_sequences(input_seqs, target_seqs, vocab_in, vocab_out):
    # Apply symbol replacement to batch
    new_input_seqs = torch.zeros_like(input_seqs)
    new_target_seqs  = torch.zeros_like(target_seqs)

    for i in range(input_seqs.shape[0]):
        input_seq, target_seq = input_seqs[0].numpy(), target_seqs[0].numpy()
        input_seq_new, target_seq_new = replace_symbols(input_seq, target_seq, vocab_in, vocab_out)
        new_input_seqs[i] = torch.tensor(input_seq_new)
        new_target_seqs[i] = torch.tensor(target_seq_new)
        
    return new_input_seqs, new_target_seqs