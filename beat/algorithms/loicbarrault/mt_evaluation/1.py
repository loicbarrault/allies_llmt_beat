# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings

import numpy as np
import sacrebleu
import logging
import pdb

beat_logger = logging.getLogger('beat_lifelong_mt')
logging.basicConfig(level=logging.DEBUG)

def compute_clipped_ngram_counts(ref, hyp, penalisation, case_sensitive):
    counts = [0,0,0,0]
    ref_len = 0
    hyp_len = 0

    if len(ref[0]) != len(hyp):
        beat_logger.error("mt_evaluation::compute_clipped_ngram_counts: ERROR: hyp ({}) and ref ({}) have different sizes.".format(len(ref), len(hyp)))

    for sent_id in range(len(hyp)):
        ref_sent = ref[0][sent_id]
        hyp_sent = hyp[sent_id]
        ref_len += len(ref_sent)
        hyp_len += len(hyp_sent)
        #TODO: compute ngrams_clip

    bleu = sacrebleu.corpus_bleu(hyp, ref, lowercase=(case_sensitive==False))

    #Apply penalisation
    beat_logger.debug('counts before {}, pen {}'.format(bleu.counts, penalisation))
    bleu.counts = [a - b for a, b in zip(bleu.counts, penalisation)]
    beat_logger.debug('bleu after {}'.format(bleu.counts))
    # NOTE: test with the reference to get a 100% BLEU! -> it works!
    #bleu = sacrebleu.corpus_bleu(ref[0], ref, lowercase=(case_sensitive==False))

    beat_logger.debug(bleu.format())
    return bleu.counts, bleu.totals, bleu.ref_len, bleu.sys_len

class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def __init__(self):
        pass


    # do initial setup work with the given parameters for the algorithm
    def setup(self, parameters):
        self.case_sensitive = parameters.get('case_sensitive', False)
        return True


    # this will be called each time the sync'd input has more data available to be processed
    def process(self, inputs, data_loaders, outputs):

        #hyp = inputs['hypothesis'].data.text
        #ref = inputs['reference'].data.text
        hyp = [ str(h) for h in inputs['hypothesis'].data.text ]
        ref = [[ str(r) for r in inputs['reference'].data.text ]]

        penalisation = inputs['penalisation'].data.value

        correct, total, ref_len, hyp_len = compute_clipped_ngram_counts(ref, hyp, penalisation, self.case_sensitive)

        outputs['correct_counts'].write({'value': correct })
        outputs['total_counts'].write({'value': total })
        outputs['ref_len'].write({'value': ref_len })
        outputs['hyp_len'].write({'value': hyp_len })
        return True
