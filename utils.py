import os
import re
import numpy as np
import torch.nn as nn
import codecs
from torch.nn import init

START_TAG = '<START>'
STOP_TAG = '<STOP>'

def create_dict(item_list):
    assert type(item_list) is list
    dict1 = {}
    for items in item_list:
        for item in items:
            if item not in dict1:
                dict1[item] = 1
            else:
                dict1[item] += 1
    return dict1

def create_mapping(object_dict):
    sorted_items = sorted(object_dict.items(), key=lambda x: (-x[1], x[0])) #sort in descending order of freq
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def zero_digits(s):
    return s
    # return re.sub('\d', '0', s)

def init_linear(input_linear):
    """Initialize linear transformation"""
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def adjust_learning_rate(optimizer, lr):
    """shrink learning rate for pytorch"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_lstm(input_lstm):
    """Initialize lstm"""
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def load_sentences(path, lower, zeros):
    sentences = []
    sentence = []
    last = False
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        word = line.split()
        if len(word) < 1:
            sentences.append(sentence)
            sentence = []
            last = True
        else:    
            sentence.append(word)
            last = False
    if not last:
        sentences.append(sentence)
    return sentences


def word_mapping(sentences, lower):
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dict(words)       # get dictionary of word-freq

    dico['pad'] = 10000001
    dico['unk'] = 10000000
    dico = {k:v for k,v in dico.items() if v>=1}    # remove words with low freq
    word_to_id, id_to_word = create_mapping(dico)

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences):
    chars = ["".join([w[0] for w in s]) for s in sentences] # join all words of  sentences into a strings
    dico = create_dict(chars)
    dico['pad'] = 10000000
    # dico[';'] = 0
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dict(tags)
    dico[START_TAG] = -1
    dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=True):
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else 'unk']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'tags': tags,       # make no sense for the test dataset :D
        })
    return data

