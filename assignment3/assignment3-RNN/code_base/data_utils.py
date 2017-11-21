from __future__ import print_function

from builtins import range
import numpy as np
import os
import platform
import pandas as pd
from nltk import download
from nltk import word_tokenize as wtk

def load_data(filename, split='train', sample=False, sample_size=100):
    data = pd.read_csv(filename, sep='\t', header=None)
    x = []
    y = []
    for i in range(len(data.values)):
        x.append(data.values[i][1])
        y.append(data.values[i][0])
    dic = {}
    if sample == False:
        dic[split] = np.array(x)
        dic[split + '_label'] = np.array(y, dtype=np.int32)
    else: # only use the first sample_size sample for training
        dic[split] = np.array(x[:sample_size])
        dic[split + '_label'] = np.array(y[:sample_size], dtype=np.int32)
    return dic

def sample_minibatch(data, batch_size=50, split='train', random=False):
    split_size = data[split].shape[0]
    if random == True:
        batch = np.random.choice(split_size, batch_size)
    else:
        batch = np.arange(batch_size)
    labels = data[split+'_label'][batch]
    sentences = data[split][batch]
    return sentences, labels

def build_dictionary(sentences, filename='code_base/datasets/dictionary.csv'):
    word_list = []
    for s in sentences:
        words = wtk(s.lower())
        for w in words:
            if w not in word_list:
                word_list.append(w)
    dic = {}
    dic['word'] = word_list
    df = pd.DataFrame(data=dic)
    df.to_csv(filename, sep='\t', header=None, index=False)

def load_dictionary(filename):
    data = pd.read_csv(filename, sep='\t', header=None)
    dic = {}
    for i in range(len(data.values)): # leave index 0 for ending of a sentence
        dic[data.values[i][0]] = i+1
    return dic

def one_hot_encoding(sentences, dictionary, max_length):
    vocab_size = len(dictionary)
    wordvecs = [] # of shape (N, T, V)
    mask = [] # of shape (N, T)
    for s in sentences:
        words = wtk(s.lower())
        tmpw = [0 for i in range(max_length)]
        tmpm = [0 for i in range(max_length)]
        for idx, w in enumerate(words):
            if idx >= max_length:
                break
            tmpw[idx] = dictionary[w]
            tmpm[idx] = 1
        one_hot = [[0 for i in range(vocab_size)] for j in range(max_length)] # (T, V)
        for i in range(len(tmpw)):
            if tmpw[i]:
                one_hot[i][tmpw[i]-1] = 1
        wordvecs.append(one_hot)
        mask.append(tmpm)
    return np.array(wordvecs), np.array(mask)

def download_corpus():
    download('punkt')

def reverse_wordvec(wordvecs, mask):
    rev = wordvecs.copy()
    N, T, _ = wordvecs.shape
    count = np.sum(mask, axis=1, dtype=np.int32)
    for i in range(N):
        for j in range(T):
            if j < count[i]:
                rev[i, j] = wordvecs[i, count[i]-j-1]
    return rev

