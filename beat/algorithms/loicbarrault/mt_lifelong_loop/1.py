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
import torch
import pickle
import struct
import copy
import logging
import random

from nmtpytorch.config import Options, TRAIN_DEFAULTS
from nmtpytorch.translator import Translator
from nmtpytorch.utils.beat import beat_separate_train_valid

from pathlib import Path

from io import BytesIO

import pdb

beat_logger = logging.getLogger('beat_lifelong_mt')
beat_logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, format='LIFELONG: %(message)s')
logging.basicConfig(level=logging.DEBUG)

TRANSLATE_DEFAULTS = {
    'suppress_unk': False,                        #Do not generate <unk> tokens.')
    'disable_filters': True,                     #Disable eval_filters given in config')
    'n_best': False,                              #Generate n-best list of beam candidates.')
    'batch_size': 16,                             #Batch size for beam-search')
    'beam_size': 6,                               #Beam size for beam-search')
    'max_len': 200,                               #Maximum sequence length to produce (Default: 200)')
    'lp_alpha': 0.,                               #Apply length-penalty (Default: 0.)')
    'device-id': 'gpu',                           #cpu or gpu (Default: gpu)')
    'models': [],                                 #Saved model/checkpoint file(s)")
    'task_id': 'src:Text -> trg:Text',            #Task to perform, e.g. en_src:Text, pt_src:Text -> fr_tgt:Text')
    'override': [],                               #(section).key:value overrides for config")
    'stochastic': False,                          #Don't fix seed for sampling-based models.")
    'beam_func': 'beam_search',                   #Use a custom beam search method defined in the model.")
    # You can translate a set of splits defined in the .conf file with -s
    # Or you can provide another input configuration with -S
    # With this, the previous -s is solely used for the naming of the output file
    # and you can only give 1 split with -s that corresponds to new input you define with -S.
    'splits': 'lifelong',                         #Comma separated splits from config file')
    'source': None,                               #Comma-separated key:value pairs to provide new inputs.')
    'output': '/not/used/because/beat_platform'   #Output filename prefix')
    }
'''
TRANSLATE_DEFAULTS = {}
'''
def mt_to_allies(hypothesis):
    hyps = {"text": hypothesis}
    return hyps


def run_translation(translator, source, file_id):
    """

    """
    beat_logger.debug("### mt_lifelong_loop:run_translation for {}".format(file_id))
    ####################################################################################
    #  Forward pass on the model
    ####################################################################################
    #TODO: prepare lifelong data to translate
    translator.update_split_source(TRANSLATE_DEFAULTS['splits'], source)

    # Run translation on the new input using mt baseline system
    hypothesis = translator()

    return hypothesis


def unsupervised_model_adaptation(source,
                  file_id,
                  data_dict,
                  model,
                  translator,
                  current_hypothesis):

    #beat_logger.debug("### mt_lifelong_loop:unsupervised_model_adaptation")

    ########################################################
    # ACCESS TRAINING DATA TO ADAPT IF WANTED
    ########################################################
    # You are allowed to re-process training data all along the lifelong adaptation.
    # The code below shows how to acces data from the training set.

    # The training and valid data is accessible through the data_dict
    # See below how to get all training data
    # The function beat_separate_train_valid allows to split the data into train/valid as during training (with loss of document information)
    data_dict_train, data_dict_valid = beat_separate_train_valid(data_dict)

    # Get the list of training files
    # We create a dictionnary with file_id as keys and the index of the file in the training set as value.
    # This dictionnary will be used later to access the training files by file_id

    ########################################################
    # Shows how to access training data per INDEX
    #
    # Shows how to access source  and file_info from
    # a training file by using its index in the training set
    ########################################################
    file_index = 2  # index of the file we want to access
    #file_id_per_index = train_loader[file_index][0]['train_file_info'].file_id
    #time_stamp_per_index = train_loader[file_index][0]['train_file_info'].time_stamp
    #supervision_per_index = train_loader[file_index][0]['train_file_info'].supervision
    #source_per_index = train_loader[file_index][0]["train_source"].value

    ########################################################
    # Shows how to access training data per file_id
    #
    # Shows how to access source text and file_info from
    # a training file by using its file_id
    ########################################################
    # Assume that we know the file_id, note that we stored them earlier in a dictionnary
    train_file_id = list(data_dict.keys())[2]
    time_stamp_per_ID = data_dict[train_file_id]["file_info"]['time_stamp']
    supervision_per_ID = data_dict[train_file_id]["file_info"]['supervision']
    source_per_ID = data_dict[train_file_id]["source"]

    # TODO: Update the model and the translator object
    
    return model, translator


def generate_system_request_to_user(model, file_id, source,  current_hypothesis):
    """
    Include here your code to ask "questions" to the human in the loop

    Prototype of this function can be adapted to your need as it is internal to this block
    but it must return a request as a dictionary object as shown below

    request_type can only be "reference" for now
        if "reference", the question asked to the user is: What is the reference translation for sentence sentence_id of the document file_id
                the cost of this question is proportional to the length of the sentence to translate

    file_id must be an existing file_id of a document in the lifelong corpus
    sentence_id must be a unsigned int between 0 and document length
    """
    # choose a sentence to be translated by the human translator
    # For now, we'll use a random sentence
    sid = random.randint(0, len(source)-1)

    #beat_logger.debug("### mt_lifelong_loop:generate_system_request_to_user")
    request = {
           "request_type": "reference",
           "file_id": '{}'.format(file_id),
           "sentence_id": np.uint32(sid)
          }

    return request


def online_adaptation(model, params, data_dict_train, file_id, doc_source, current_hypothesis):
    """
    Include here your code to adapt the model according
    to the human domain expert answer to the request

    Prototype of this function can be adapted to your need as it is internatl to this block
    """
    beat_logger.debug("### mt_lifelong_loop:online_adaptation")
    #print("### mt_lifelong_loop:online_adaptation")

    # Case of active learning
    new_hypothesis = current_hypothesis

    return model, new_hypothesis


class Algorithm:
    """
    Main class of the mt_lifelong_loop
    """
    def __init__(self):
        beat_logger.debug("### mt_lifelong_loop:init -- that's great!")
        self.model = None
        self.translator = None
        self.data_dict_train = None
        self.data_dict_dev = None
        self.translate_params=TRANSLATE_DEFAULTS
        self.init_end_index = -1

    def setup(self, parameters):

        self.params={}
        self.params['train']=TRAIN_DEFAULTS
        self.params['model']={}

        self.params['train']['seed']=int(parameters['seed'])
        self.params['train']['model_type']=parameters['model_type']
        self.params['train']['patience']=int(parameters['patience'])
        self.params['train']['max_epochs']=2  #int(parameters['max_epochs'])
        self.params['train']['eval_freq']=int(parameters['eval_freq'])
        #self.params['train']['eval_metrics']=parameters['eval_metrics']
        self.params['train']['eval_filters']=parameters['eval_filters']
        self.params['train']['eval_beam']=int(parameters['eval_beam'])
        self.params['train']['eval_batch_size']=int(parameters['eval_batch_size'])
        self.params['train']['save_best_metrics']=parameters['save_best_metrics']
        self.params['train']['eval_max_len']=int(parameters['eval_max_len'])
        self.params['train']['checkpoint_freq']=int(parameters['checkpoint_freq'])
        #self.params['train']['n_checkpoints']=parameters['n_checkpoints']
        self.params['train']['l2_reg']=int(parameters['l2_reg'])
        self.params['train']['lr_decay']=parameters['lr_decay']
        self.params['train']['lr_decay_revert']=parameters['lr_decay_revert']
        self.params['train']['lr_decay_factor']=parameters['lr_decay_factor']
        self.params['train']['lr_decay_patience']=int(parameters['lr_decay_patience'])
        self.params['train']['gclip']=int(parameters['gclip'])
        self.params['train']['optimizer']=parameters['optimizer']
        self.params['train']['lr']=parameters['lr']
        self.params['train']['batch_size']=int(parameters['batch_size'])
        self.params['train']['save_optim_state']=False

        self.params['train']['save_path']=Path("/not/used/because/beat_platform")
        self.params['train']['tensorboard_dir']="/lium/users/barrault/llmt/tensorboard"

        self.params['model']['att_type']=parameters['att_type']
        self.params['model']['att_bottleneck']=parameters['att_bottleneck']
        self.params['model']['enc_dim']=int(parameters['enc_dim'])
        self.params['model']['dec_dim']=int(parameters['dec_dim'])
        self.params['model']['emb_dim']=int(parameters['emb_dim'])
        self.params['model']['dropout_emb']=parameters['dropout_emb']
        self.params['model']['dropout_ctx']=parameters['dropout_ctx']
        self.params['model']['dropout_out']=parameters['dropout_out']
        self.params['model']['n_encoders']=int(parameters['n_encoders'])
        self.params['model']['tied_emb']=parameters['tied_emb']
        self.params['model']['dec_init']=parameters['dec_init']
        self.params['model']['bucket_by']="src"
        if parameters['max_len']=="None":
            self.params['model']['max_len']=None
        else:
            self.params['model']['max_len']=int(parameters['max_len'])
        self.params['model']['direction']="src:Text -> trg:Text"

        return True

    def process(self, inputs, data_loaders, outputs, loop_channel):
        #print("### mt_lifelong_loop:process start -- that's great!")
        """

        """

        ########################################################
        # RECEIVE INCOMING INFORMATION FROM THE FILE TO PROCESS
        ########################################################

        # Access source text of the current file to process
        source = inputs["processor_lifelong_source"].data.text

        if self.translator is None or self.train_data is None:
            # Get the model after initial training
            dl = data_loaders[0]
            (data, _,end_index) = dl[0]

            # Create the nmtpy Translator object to translate
            if self.translator is None:
                model_data = data['model'].value
                model_data = struct.pack('{}B'.format(len(model_data)), *list(model_data))
                # Create a Translator object from nmtpy
                self.translate_params['models'] = [model_data]
                self.translate_params['source'] = source
                self.translator = Translator(beat_platform=True, **self.translate_params)

            if self.data_dict_train is None:
                self.data_dict = pickle.loads(data["processor_train_data"].text.encode("latin1"))
                self.data_dict_train, self.data_dict_dev = beat_separate_train_valid(self.data_dict)

        # Access incoming file information
        # See documentation for a detailed description of the mt_file_info
        file_info = inputs["processor_lifelong_file_info"].data
        file_id = file_info.file_id
        supervision = file_info.supervision
        time_stamp = file_info.time_stamp

        beat_logger.debug("mt_lifelong_loop::process: received document {} ({} sentences) to translate ".format(file_id, len(source)))
        #print('mt_lifelong_loop::process: source = {}'.format(source))


        #TODO: prepare train/valid data for fine-tuning (eventually) -- might not be needed actually as the data is already contained in the training params -> this can be huge!
        current_hypothesis = run_translation(self.translator, source, file_id)
        # If human assisted learning is ON
        ###################################################################################################
        # Interact with the human if necessary
        # This section exchange information with the user simulation and ends up with a new hypothesis
        ###################################################################################################
        human_assisted_learning = supervision in ["active", "interactive"]

        if not human_assisted_learning:
            # In this method, see how to access initial training data to adapt the model
            # for the new incoming data
            self.model, self.translator = unsupervised_model_adaptation(source,
                                         file_id,
                                         self.train_data,
                                         self.model,
                                         self.translator,
                                         current_hypothesis
                                         )
            # update current_hypothesis with current model
            current_hypothesis = run_translation(self.translator, source, file_id)


        # If human assisted learning mode is on (active or interactive learning)
        while human_assisted_learning:

            # Create an empty request that is used to initiate interactive learning
            # For the case of active learning, this request is overwritten by your system itself
            # For now, only requests of type 'reference' are allowed (i.e. give me the reference translation for sentence 'sentence_id' of file 'file_id')


            if supervision == "active":
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system
                request = generate_system_request_to_user(self.model, file_id, source, current_hypothesis)

                # Send the request to the user and wait for the answer
                message_to_user = {
                    "file_id": file_id,  # ID of the file the question is related to
                    "hypothesis": current_hypothesis[request['sentence_id']] ,  # The current hypothesis
                    "system_request": request,  # the question for the human in the loop
                }
                #beat_logger.debug("mt_lifelong_loop::process: send message to user: request={}".format(request))
                human_assisted_learning, user_answer = loop_channel.validate(message_to_user)

                # Take into account the user answer to generate a new hypothesis and possibly update the model
                self.model = online_adaptation(self.model, self.params, self.data_dict_train, file_id, source, current_hypothesis)

                # Update the translator object with the current model
                self.translate_params['models'] = [self.model]
                self.translator = Translator(beat_platform=True, **self.translate_params)

                # Generate a new translation
                new_hypothesis = run_translation(translator, source, file_id)
                #new_hypothesis = current_hypothesis

                beat_logger.debug("BEFORE online_adaptation: {}".format(current_hypothesis))
                beat_logger.debug("AFTER online_adaptation : {}".format(new_hypothesis))


            else:
                human_assisted_learning = False

        # End of human assisted learning
        # Send the current hypothesis
        #    self.init_end_index = 0
        #beat_logger.debug("HYPOTHESIS: {}".format(current_hypothesis))
        #print("mt_lifelong_loop::process: translated document {}: ".format(file_id), current_hypothesis)
        outputs["hypothesis"].write(mt_to_allies(current_hypothesis))

        if not inputs.hasMoreData():
            pass
        # always return True, it signals BEAT to continue processing
        return True



