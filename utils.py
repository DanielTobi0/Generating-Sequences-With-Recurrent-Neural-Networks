import string
import unicodedata
import torch


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def load_data(filename):
    with open(filename, encoding='utf-8') as file:
        return ' '.join([unicodeToAscii(line.strip()) for line in file])

def inputTensor(line, n_letters, all_letters):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor