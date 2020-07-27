# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings
import numpy as np
import pickle
import pdb
import logging
import json
import io

from nmtpytorch.utils.beat  import create_vocab
import mosestokenizer as mtok

import subword_nmt as bpe
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE, read_vocabulary
from subword_nmt.get_vocab import get_vocab


beat_logger = logging.getLogger('beat_lifelong_mt')
#beat_logger.setLevel(logging.DEBUG)
#beat_logger.setLevel(logging.CRITICAL)         #logging.CRITICAL logging.ERROR logging.WARNING logging.INFO logging.DEBUG

logging.basicConfig(level=logging.INFO, format='%(message)s')


def setup_for_nmtpytorch(data_loaders):
    data_dict = {}
    data_loader = data_loaders[0]
    # for each document, get source and target text and index this
    for i in range(data_loader.count()):
        (data, _, end_index) = data_loader[i]
        doc_dict = {}
        doc_dict["source"] = data["train_source_raw"].text
        doc_dict["target"] = data["train_target_raw"].text
        doc_dict["file_info"] = {}
        doc_dict["file_info"]["supervision"] = data["train_file_info"].supervision
        doc_dict["file_info"]["time_stamp"] = data["train_file_info"].time_stamp
        doc_dict["file_info"]["in_dev"] = data["train_file_info"].in_dev
        file_id = data["train_file_info"].file_id
        data_dict[file_id] = doc_dict

    return data_dict, end_index

def train_subword_model(src_text, trg_text, nb_symbols=10000):

    # create text content with source and target text
    content = []
    content.extend(src_text)
    content.extend(trg_text)

    bpe_model_io = io.StringIO()
    src_vocab_io = io.StringIO()
    trg_vocab_io = io.StringIO()

    # 1. Learn BPE model on both source and target text
    # 1.1 cat {train_file}.L1 {train_file}.L2 | subword-nmt learn-bpe -s {num_operations} -o {codes_file}
    # 1.2 subword-nmt apply-bpe -c {codes_file} < {train_file}.L1 | subword-nmt get-vocab > {vocab_file}.L1
    # 1.3 subword-nmt apply-bpe -c {codes_file} < {train_file}.L2 | subword-nmt get-vocab > {vocab_file}.L2

    # 1.1 learn_bpe(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input, total_symbols=args.total_symbols))
    learn_bpe(content, bpe_model_io, nb_symbols, 0, False, False, False)

    # 1.2
    src_text_tok = apply_bpe(bpe_model_io, src_text, merges=nb_symbols)
    get_vocab(src_text_tok, src_vocab_io)
    src_vocab = read_vocabulary(src_vocab_io, 0)
    # 1.3
    trg_text_tok = apply_bpe(bpe_model_io, trg_text, merges=nb_symbols)
    get_vocab(trg_text_tok, trg_vocab_io)
    trg_vocab = read_vocabulary(trg_vocab_io, 0)

    # 3. Re-apply BPE with the obtained vocabulary
    # subword-nmt apply-bpe -c {codes_file} --vocabulary {vocab_file}.L1 --vocabulary-threshold 50 < {train_file}.L1 > {train_file}.BPE.L1
    src_text_tok = apply_bpe(bpe_model_io, src_text, vocab=src_vocab)
    trg_text_tok = apply_bpe(bpe_model_io, trg_text, vocab=trg_vocab)

    bpe_model = bpe_model_io.getvalue()

    bpe_model_io.close()
    src_vocab_io.close()
    trg_vocab_io.close()

    return bpe_model, src_text_tok, trg_text_tok

"""

"""
def apply_bpe(bpe_model, text, merges=-1, vocab=None):
    #TODO: do something here
    bpetext = []
    bpe = BPE(bpe_model, merges=merges, vocab=vocab)
    for line in text:
        bpetext.append(bpe.process_line(line))
    return bpetext


"""
Tokenise with Moses tokenizer
Uses https://github.com/luismsgomes/mosestokenizer
"""
def preprocess(data_dict, src_lang, trg_lang, min_freq=0, short_list=0, max_sent_len=-1):
    src_text = []
    trg_text = []

    src_tokenizer = mtok.MosesTokenizer(src_lang)
    trg_tokenizer = mtok.MosesTokenizer(trg_lang)

    # tokenize data
    for doc in data_dict:
        for i, t in enumerate(data_dict[doc]["source"]):
            data_dict[doc]["source"][i] = ' '.join(src_tokenizer(t))
            src_text.append(data_dict[doc]["source"][i])
        for i, t in enumerate(data_dict[doc]["target"]):
            data_dict[doc]["target"][i] = ' '.join(trg_tokenizer(t))
            trg_text.append(data_dict[doc]["target"][i])
    src_tokenizer.close()
    trg_tokenizer.close()
    # train subword model
    subword_model, src_text, trg_text = train_subword_model(src_text, trg_text, nb_symbols=short_list)

    # Re-create the vocabulary with the nmtpy format
    src_vocab = create_vocab(src_text, min_freq, short_list)
    trg_vocab = create_vocab(trg_text, min_freq, short_list)

    #NOTE: put back processed text into data dict
    for doc in data_dict:
        nbsent = len(data_dict[doc]["source"])
        data_dict[doc]['source'] = []
        data_dict[doc]['target'] = []
        for i in range(nbsent):
            src = src_text.pop(0)
            trg = trg_text.pop(0)
            if len(src.split(' ')) <=  max_sent_len and len(trg.split(' ')) <= max_sent_len:
                data_dict[doc]['source'].append(src)
                data_dict[doc]['target'].append(trg)

        if len(data_dict[doc]['source']) == 0:
            print("Document {} contains only lengthy sentences, I'm removing it...".format(doc))
            del data_dict[doc]

    assert len(src_text) == 0 , "preprocess: something is wrong with source text"
    assert len(trg_text) == 0 , "preprocess: something is wrong with target text"

    return data_dict, src_vocab, trg_vocab, subword_model



class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def setup(self, parameters):
        # Retrieve the value of the parameters
        self.source_language = parameters['source_language']
        self.target_language = parameters['target_language']
        self.min_freq = parameters['min_freq']
        self.short_list = parameters['short_list']
        self.max_sent_len = parameters['max_sent_len']
        return True


    # this will be called each time the sync'd input has more data available to be processed
    def process(self, data_loaders, outputs):
        beat_logger.debug("mt_train_preprocessing: SRC LANG = {} TRG LANG = {}".format(self.source_language, self.target_language))
        beat_logger.debug("mt_train_preprocessing::setup: min_freq = {} short_list = {}".format(self.min_freq, self.short_list))

        data_dict, end_index = setup_for_nmtpytorch(data_loaders)

        #NOTE: create vocabulary and BPE or SPM model
        data_dict_tok, src_vocab, trg_vocab, subword_model = preprocess(data_dict, self.source_language, self.target_language, self.min_freq, self.short_list, self.max_sent_len)

        data_dict_pickle = pickle.dumps(data_dict_tok).decode("latin1")

        outputs['train_data_tokenized'].write({'text':data_dict_pickle}, end_index)
        outputs['source_vocabulary'].write({'text':src_vocab}, end_index)
        outputs['target_vocabulary'].write({'text':trg_vocab}, end_index)
        outputs['subword_model'].write({'text':subword_model}, end_index)
        beat_logger.debug("############## End of mt_train_preprocessing ############")

        # always return True, it signals BEAT to continue processing
        return True







