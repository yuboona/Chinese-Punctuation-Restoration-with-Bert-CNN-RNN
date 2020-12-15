#-*- coding : utf-8-*-
# coding:unicode_escape
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    return data

def encode_data(data, tokenizer, punctuation_enc):
    """
    Converts words to (BERT) tokens and puntuation to given encoding.
    Note that words can be composed of multiple tokens.
    """
    X = []
    Y = []
    for line in data:
        word, punc = line.split()

        punc = punc.strip()
        tokens = tokenizer.tokenize(word)
        x = tokenizer.convert_tokens_to_ids(tokens)
        y = [punctuation_enc[punc]]
        if len(x) > 0:
            if len(x) > 1:
                y = (len(x)-1)*[0]+y
            X += x
            Y += y
    return X, Y


def preprocess_data(data, tokenizer, punctuation_enc, seq_len):
    X, Y = encode_data(data, tokenizer, punctuation_enc)
    length = len(X)
    X = np.array(X)
    Y = np.array(Y)
    remain = length % seq_len
    X = X[:-remain].reshape((-1, seq_len))
    Y = Y[:-remain].reshape((-1, seq_len))
    # print('X shape', X.shape)
    return X, Y

def create_data_loader(X, y, shuffle, batch_size):
    data_set = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(np.array(y)).long())
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader