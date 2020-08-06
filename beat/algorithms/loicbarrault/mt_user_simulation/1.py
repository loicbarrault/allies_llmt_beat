#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

###################################################################################
#                                                                                 #
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/               #
# Contact: beat.support@idiap.ch                                                  #
#                                                                                 #
# Redistribution and use in source and binary forms, with or without              #
# modification, are permitted provided that the following conditions are met:     #
#                                                                                 #
# 1. Redistributions of source code must retain the above copyright notice, this  #
# list of conditions and the following disclaimer.                                #
#                                                                                 #
# 2. Redistributions in binary form must reproduce the above copyright notice,    #
# this list of conditions and the following disclaimer in the documentation       #
# and/or other materials provided with the distribution.                          #
#                                                                                 #
# 3. Neither the name of the copyright holder nor the names of its contributors   #
# may be used to endorse or promote products derived from this software without   #
# specific prior written permission.                                              #
#                                                                                 #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED   #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE          #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE    #
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL      #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR      #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE   #
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.            #
#                                                                                 #
###################################################################################

import numpy as np
import pdb
import sys
import logging
from sacrebleu.sacrebleu import NGRAM_ORDER

beat_logger = logging.getLogger('beat_lifelong_mt')
beat_logger.setLevel(logging.DEBUG)


def get_document_length(doc):
    nbw = 0
    for sent in doc:
        #print('doclen: sent=#{}# len={}'.format(sent.replace('@@ ', '').split(' '), len(sent.replace('@@ ', '').split(' '))))
        nbw += len(sent.replace('@@ ', '').split(' '))
    return nbw




class Algorithm:
    def __init__(self):
        pass

    def setup(self, parameters):
        beat_logger.debug("### mt_user_simulation:setup ")
        self.cost = 0
        self.penalisation = [0]*NGRAM_ORDER
        beat_logger.debug("mt_user_simulation::init: params {}".format(parameters))
        self.max_cost_per_file = parameters["max_cost_per_file"]
        return True

    def update_penalisation(self, answer):
        #beat_logger.debug("### mt_user_simulation:update_penalisation ")
        #TODO: compute cost of providing the reference for this sentence
        #NOTE: This cost will  be used to calculate the BLEU penalisation

        #print("user_simulation::update_penalisation: answer = {}".format(answer))

        if answer["response_type"] == "reference":
            answer_words = answer['answer'].replace('@@ ', '').split()
            nb_words = len(answer_words)
            sent_pen = [ nb_words-x for x in range(NGRAM_ORDER) ]
            new_pen = [ x + y for x, y in zip(self.penalisation, sent_pen )]
            print("Penalisation was {}, len(ref) is {}, new penalisation if {}".format(self.penalisation, nb_words, new_pen))
            #print("update penalisation: ref='{}'. length={}".format(' '.join(answer_words), nb_words))
            self.penalisation = new_pen
            return nb_words
        return 0

    def validate(self, request):
        #beat_logger.info("### mt_user_simulation:validate ")
        beat_logger.info("mt_user_simulation::validate: file_id={} req.file_id={} system_request={}".format(self.file_info.file_id, request.file_id, request.system_request))
        #beat_logger.debug("mt_user_simulation::validate: max_cost_per_doc = {} current cost = {}".format(self.max_cost_per_file, self.cost))

        file_id = self.file_info.file_id
        sent_id = request.system_request.sentence_id

        answer = {}
        if self.cost >= self.max_cost:
            #beat_logger.debug( "mt_user_simulation::validate: System reach maximum cost! ## sentence {} of file {}".format(sent_id, self.file_info.file_id))
            #print( "mt_user_simulation::validate: System reach maximum cost! ## sentence {} of file {}".format(sent_id, self.file_info.file_id))
            answer = {
                "answer": self.reference.text[sent_id],
                "response_type": "stop",
                "file_id": self.file_info.file_id,
                "sentence_id": sent_id
                }

        elif self.file_info.supervision == "active":
            beat_logger.debug( "mt_user_simulation::active: System ask reference for sentence {} of file {}".format(sent_id, self.file_info.file_id))
            print( "#########################  mt_user_simulation::active: System ask reference for sentence {} of file {}".format(sent_id, self.file_info.file_id))
            answer = {
                "answer": self.reference.text[sent_id],
                "response_type": "reference",
                "file_id": self.file_info.file_id,
                "sentence_id": sent_id
                }

        else: #if self.file_info.supervision == "interactive":
            beat_logger.debug( "mt_user_simulation::interactive: human provided reference for sentence {} of file {}".format(sent_id, self.file_info.file_id))
            #print( "############################# mt_user_simulation::interactive: human provided reference for sentence {} of file {}".format(sent_id, self.file_info.file_id))
            answer = {
                "answer": self.reference.text[sent_id],
                "response_type": "stop",
                "file_id": self.file_info.file_id,
                "sentence_id": sent_id
                }

        self.cost += self.update_penalisation(answer)
        return ( answer['response_type'] != "stop", answer )

    def write(self, outputs, processor_output_name, end_data_index):
        #beat_logger.debug("### mt_user_simulation:write ")
        #print( "user::write file={} cost={} penalisation={} end_data_index={}".format(self.file_info.file_id, self.cost, self.penalisation, end_data_index))
        outputs["evaluator_output"].write({"value": self.penalisation}, end_data_index)
        return True

    def read(self, inputs):
        #beat_logger.debug("### mt_user_simulation:read ")
        self.file_info = inputs["evaluator_file_info"].data
        self.source = inputs["evaluator_source"].data
        self.reference = inputs["evaluator_target"].data
        self.doc_len = get_document_length(self.reference.text)

        self.max_cost = self.max_cost_per_file * self.doc_len / 100.0
        beat_logger.debug('max cost = {}'.format(self.max_cost))

        self.penalisation = [0,0,0,0]
        self.cost = 0
        #beat_logger.debug( "user::read file {} supervision {} ".format(self.file_info.file_id, self.file_info.supervision))
        return True

    def process(self, inputs, data_loaders, outputs, channel):
        #beat_logger.debug("### mt_user_simulation:process start", file=sys.stderr)
        #beat_logger.debug("### mt_user_simulation:process end", file=sys.stderr)
        pass


