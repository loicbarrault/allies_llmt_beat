# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings
import numpy
import logging
import mosestokenizer as mtok
import io
import pickle

from subword_nmt.apply_bpe import BPE

import ipdb
beat_logger = logging.getLogger('beat_lifelong_mt')

"""
Tokenise with Moses tokenizer
Uses https://github.com/luismsgomes/mosestokenizer
"""
def preprocess(document, tokenizer, subword_model):

    level = logging.root.level
    logging.basicConfig(level=logging.ERROR, format='%(message)s')

    #TODO: tokenize data
    #TODO: apply subword model
    for i, sent in enumerate(document):
        document[i] = subword_model.process_line(' '.join(tokenizer(sent)))

    logging.basicConfig(level=level, format='%(message)s')
    return document

class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def setup(self, parameters):
        # Retrieve the value of the parameters
        self.source_language = parameters['source_language']
        self.target_language = parameters['target_language']
        self.src_tokenizer = mtok.MosesTokenizer(self.source_language)
        self.trg_tokenizer = mtok.MosesTokenizer(self.target_language)
        self.src_bpe = None
        self.trg_bpe = None
        self.src_vocab = None
        self.trg_vocab = None
        return True

    # this will be called each time the sync'd input has more data available to be processed
    def process(self, inputs, data_loaders, outputs):

        #beat_logger.debug("mt_apply_preprocessing: SRC LANG = {} TRG LANG = {}".format(self.source_language, self.target_language))
        if self.src_bpe is None or self.trg_bpe is None or self.src_vocab is None or self.trg_vocab is None:
            (data, _, end_data_index) = data_loaders[0][0]
            if self.src_vocab is None:
                self.src_vocab = pickle.loads(data["subword_source_vocabulary"].text.encode("latin1"))
            if self.trg_vocab is None:
                self.trg_vocab = pickle.loads(data["subword_target_vocabulary"].text.encode("latin1"))

            subword_model = io.StringIO(data["subword_model"].text)
            if self.src_bpe is None:
                self.src_bpe = BPE(subword_model, vocab=self.src_vocab)
            if self.trg_bpe is None:
                self.trg_bpe = BPE(subword_model, vocab=self.trg_vocab)

        lifelong_source_raw = []
        for s in inputs['lifelong_source_raw'].data.text:
            lifelong_source_raw.append(s)

        lifelong_target_raw = []
        for s in inputs['lifelong_target_raw'].data.text:
            lifelong_target_raw.append(s)

        #TODO: apply subword model to the lifelong data 
        # but for now, let's just pass the text as is
        lifelong_source_tok = preprocess(lifelong_source_raw, self.src_tokenizer, self.src_bpe )
        lifelong_target_tok = preprocess(lifelong_target_raw, self.trg_tokenizer, self.trg_bpe )

        outputs['lifelong_source_tokenized'].write({'text':lifelong_source_tok})
        outputs['lifelong_target_tokenized'].write({'text':lifelong_target_tok})
        #print("{}\n".format(lifelong_source_tok))
        #print("{}\n".format(lifelong_target_tok))

        #beat_logger.debug("############## End of mt_apply_preprocessing ############")
        # always return True, it signals BEAT to continue processing
        if not inputs.hasMoreData():
            self.src_tokenizer.close()
            self.trg_tokenizer.close()

        return True







