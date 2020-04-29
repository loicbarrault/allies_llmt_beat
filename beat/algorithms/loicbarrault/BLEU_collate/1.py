# You may import any python packages that will be available in the environment you will run this algorithm in
# Environments can change based on the experiment's settings

import numpy as np
import sacrebleu
import logging
import pdb


beat_logger = logging.getLogger('beat_lifelong_mt')

def compute_bleu(correct, total, hyp_len, ref_len):
    bleu = sacrebleu.compute_bleu(correct, total, hyp_len, ref_len)
    #bleu = 0.5
    return bleu.score


class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def __init__(self):
        self.correct_counts = [0,0,0,0]
        self.total_counts = [0,0,0,0]
        self.total_ref_len = 0.0
        self.total_hyp_len = 0.0

    # this will be called each time the sync'd input has more data available to be processed
    def process(self, inputs, data_loaders, output):
        # Groups available:
        # Group Lifelong:
        #   Input "correct_counts" with type  "system/array_1d_uint32/1"
        #   Input "total_counts" with type  "anthony_larcher/array_1d_uint32/1"
        #   Input "ref_len" with type "system/int64/1"
        #   Input "hyp_len" with type "system/int64/1"
     
        # Results available:
        #   Result "BLEU" with type "float32"
     
        # to check if there is more data waiting in the inputs
        # (if it is False, you have processed all the inputs and this "process" function won't be called again):
        #       if inputs.hasMoreData():

        # to check if a specific input is done:
        #       if inputs["input1"].isDataUnitDone():

        # to manually fetch the next input of a specific input
        # (e.g. the block is not sync'd to the input but you want the input immediately)
        #       inputs['input1'].next()
        # you can then access that input value as normal:
        #       self.val1 = inputs['input1'].data

        # to get the data for an input (note that the value will be of the type specified in the metadata!):
        #       data_value = inputs['input1'].data

        # to write to an output:
        #       outputs['output1'].write({
        #           'output_field_1': 1,
        #           'output_field_2': 'output'
        #       })

        #beat_logger.debug("### BLEU_collate::process: start")

        correct_counts = inputs['correct_counts'].data.value
        total_counts = inputs['total_counts'].data.value
        for o in range(len(correct_counts)):
            self.correct_counts[o] += correct_counts[o]
            self.total_counts[o] += total_counts[o]

        hyp_len = inputs['hyp_len'].data.value
        ref_len = inputs['ref_len'].data.value

        self.total_ref_len += ref_len
        self.total_hyp_len += hyp_len

        if not inputs.hasMoreData():
            beat_logger.info("### BLEU_collate::process: total statistics are: CORRECT: {} TOTAL: {} HYP_LEN: {} REF_LEN: {} ".format(self.correct_counts, self.total_counts, self.total_hyp_len, self.total_ref_len))
            print("### BLEU_collate::process: total statistics are: CORRECT: {} TOTAL: {} HYP_LEN: {} REF_LEN: {} ".format(self.correct_counts, self.total_counts, self.total_hyp_len, self.total_ref_len))
            bleu = compute_bleu(self.correct_counts, self.total_counts, self.total_hyp_len, self.total_ref_len)
            output.write({'BLEU': np.cast['float32'](bleu)})

        #beat_logger.debug("### BLEU_collate::process: end ")
        # always return True, it signals BEAT to continue processing
        return True
