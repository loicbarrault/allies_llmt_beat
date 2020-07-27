# -*- coding: utf-8 -*-

"""
Some utility functions for nmtpytorch to work on BEAT platform

"""
import logging
import json

from ..vocabulary import get_freqs, freqs_to_dict


logger = logging.getLogger('nmtpytorch')


"""
Separate the data into train and dev
each corpus is a dict with 2 keys 'src' and 'trg', each containing an array of sentences (str)
"""
def beat_separate_train_valid(data_dict):
    data_dict_train = {}
    data_dict_train['src'] = []
    data_dict_train['trg'] = []
    data_dict_dev = {}
    data_dict_dev['src'] = []
    data_dict_dev['trg'] = []
    nb_doc_dev = 0
    nb_sent_dev = 0
    nb_doc_train = 0
    nb_sent_train = 0
    for fid in data_dict:
        if data_dict[fid]['file_info']['in_dev']:
            nb_doc_dev += 1
            for s in data_dict[fid]['source']:
                nb_sent_dev += 1
                data_dict_dev['src'].append(s)
            for s in data_dict[fid]['target']:
                data_dict_dev['trg'].append(s)
        else:
            nb_doc_train += 1
            for s in data_dict[fid]['source']:
                nb_sent_train += 1
                data_dict_train['src'].append(s)
            for s in data_dict[fid]['target']:
                data_dict_train['trg'].append(s)

    #print("################################# DATA STATS: train len = {} doc and {} sents".format(nb_doc_train, nb_sent_train))
    #print("################################# DATA STATS: valid len = {} docs = {} sents".format(nb_doc_dev, nb_sent_dev))

    return data_dict_train, data_dict_dev


def create_vocab(text, min_freq=0, short_list=0):
    # Get frequencies
    freqs = get_freqs(text)
    # Build dictionary from frequencies
    vocab = freqs_to_dict(freqs, min_freq, short_list)
    vocab = json.dumps(vocab, ensure_ascii=False, indent=2)
    #logger.debug("nmtpytorch::create_vocab vocab len = {}".format(len(vocab)))
    return vocab



