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
import platform
import json
import os

import nmtpytorch
from nmtpytorch.config import Options, TRAIN_DEFAULTS
from nmtpytorch.mainloop import MainLoop
from nmtpytorch.translator import Translator
from nmtpytorch.utils.beat import beat_separate_train_valid
from nmtpytorch.utils.misc import setup_experiment, fix_seed
from nmtpytorch.utils.device import DeviceManager
from nmtpytorch import models

from pathlib import Path

from io import BytesIO

import kiwi
import statistics

import ipdb

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

def get_worst_translation(source, current_hypothesis, qe_model):
    worst_translation = {'index_sen':-1, 'prediction':0.0}
    # loop over all sentences in the file
    for index_sen, sen in enumerate(source):
        # for each sentence predict score - the sentence score is a prediction of the sentence's HTER (Human-targeted Translation Error Rate). Or in other words, what is the percentage of the sentence that you would need to change to create a correct translation.
        source_list = list(sen.split())
        target_list = list(current_hypothesis[index_sen].split())
        if not target_list:
            target_list.append('@')
        examples = {'source':source_list, 'target':target_list}
        predictions = qe_model.predict(examples)
        avg_translation_score = statistics.mean(predictions['sentence_scores'])
        if worst_translation['prediction'] <= avg_translation_score:
            worst_translation.update({'index_sen':index_sen,'prediction':avg_translation_score})

    return worst_translation['index_sen']

def generate_system_request_to_user(file_id, source, current_hypothesis, qe_model):
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
    if qe_model is None:        
        # For now, we'll use a random sentence
        sid = random.randint(0, len(source)-1)
    else:
        # the worst translated sentence selected by OpenKiwi
        sid = get_worst_translation(source, current_hypothesis, qe_model)
    
    #beat_logger.debug("### mt_lifelong_loop:generate_system_request_to_user")
    request = {
           "request_type": "reference",
           "file_id": '{}'.format(file_id),
           "sentence_id": np.uint32(sid)
          }

    return request

def prepare_model_for_tuning(model_data, params, data_dict_tune):

    # set the params similarly to train_model
    # add the pretrained options

    params['train']['pretrained_file'] = model_data
    #params['train']['eval_zero'] = True
    params['train']['pretrained_layers'] = ''   # comma sep. list of layer prefixes to initialize
    params['train']['freeze_layers'] = ''       # comma sep. list of layer prefixes to freeze

    # Prepare tuning and valid data
    params['data']={}
    params['data']['train_set']={}
    params['data']['train_set']['src']=data_dict_tune['src']
    params['data']['train_set']['trg']=data_dict_tune['trg']
    #NOTE: No need to valid for finetuning, simply run several epochs
    #self.params['data']['val_set']={}
    #self.params['data']['val_set']['src']=self.valid_data['src']
    #self.params['data']['val_set']['trg']=valid_data['trg']


    #get the vocabularies from the model (need to load it with torch then check data['opts']['vocabulary']['src'] / ['trg']
    in_stream = BytesIO(model_data)
    model = torch.load(in_stream)

    params['vocabulary']={}
    params['vocabulary']['src']=model['opts']['vocabulary']['src']
    params['vocabulary']['trg']=model['opts']['vocabulary']['trg']
    params['filename']='/not/needed/beat_platform'
    params['sections']=['train', 'model', 'data', 'vocabulary']

    opts = Options.from_dict(params,{})

    setup_experiment(opts, beat_platform=True)
    dev_mgr = DeviceManager("gpu")

    # If given, seed that; if not generate a random seed and print it
    if opts.train['seed'] > 0:
        seed = fix_seed(opts.train['seed'])
    else:
        opts.train['seed'] = fix_seed()

    # Instantiate the model object
    adapt_model = getattr(models, opts.train['model_type'])(opts=opts, beat_platform=True)

    beat_logger.info("Python {} -- torch {} with CUDA {} (on machine '{}')".format(
        platform.python_version(), torch.__version__,
        torch.version.cuda, platform.node()))
    beat_logger.info("nmtpytorch {}".format(nmtpytorch.__version__))
    beat_logger.info(dev_mgr)
    beat_logger.info("Seed for further reproducibility: {}".format(opts.train['seed']))

    loop = MainLoop(adapt_model, opts.train, dev_mgr, beat_platform = True)

    return loop



def get_sen_vecs(data_dict_train, src_vocab, word_embs):
    """Getting sentence vectors for all training data 

    Args:
        data_dict_train: training data with source and target language sentences, dict dtype
        src_vocab: a vocab dictionary loaded by 
            #src_vocab = json.loads(opts['vocabulary']['src']), dict dtype
        word_embs: word embeddings with dim (vocab_size,), 1-d torch.tensor dtpye

    Returns:
        A matrix of which each row represent a sentence in data_dict_train['src'], torch.tensor dtype
    """

    sen_vecs = torch.zeros(len(data_dict_train['src']), word_embs.size()[1], 
        dtype=word_embs.dtype, device=word_embs.device)

    for index_sen, sen in enumerate(data_dict_train['src']):
        splitsen = sen.split()
        sen_length = 0
        for token in splitsen:
            if token in src_vocab.keys():
                sen_length += 1
                index_token = int(src_vocab[token].split()[0])
                sen_vecs[index_sen] += word_embs[index_token]
        sen_vecs[index_sen] /= sen_length

    return sen_vecs


def data_selection_emb(N, data_dict_train, train_sen_vecs, doc_input, src_vocab, word_embs):
    """Selecting topN sentences based on average embedding cosine similarity

    Args:
        N: top N sentence that want to get
        data_dict_train: dictionnary containing training data to select from
        doc_input: an input document for data selection, dtype: dict of list of str (keys are 'src' and 'trg')
        train_sen_vecs: a matrix of which each row represent a sentence in data_dict_train['src'], torch.tensor dtype
        src_vocab: a vocabs dictionary loaded by
            #src_vocabs = json.loads(opts['vocabulary']['src']), dict dtype
        word_embs: word embeddings with dim (vocab_size,), 1-d torch.tensor dtype

    Returns:
        Dictionary containing the selected parallel sentences, dtype: dict of list of str (keys are 'src' and 'trg')
    """
    n = round(N/len(doc_input)) if N/len(doc_input) >= 1 else 1
    n = min(n, train_sen_vecs.shape[0])

    cos = torch.nn.CosineSimilarity()
    sens_selected = {}
    sens_selected['src'] = []
    sens_selected['trg'] = []

    for sen in doc_input:
        sen_vec = torch.zeros(1, word_embs.size()[1], dtype=word_embs.dtype, device=word_embs.device)
        splitsen = sen.split()
        sen_length = 0
        for token in splitsen:
            if token in src_vocab.keys():
                sen_length += 1
                index = int(src_vocab[token].split()[0])
                sen_vec += word_embs[index]
        sen_vec /= sen_length
        coss = cos(sen_vec, train_sen_vecs)
        coss[torch.isnan(coss)] = 0
        index_topn_sens = torch.topk(coss, n, largest=True).indices
        for i in index_topn_sens:
            sens_selected['src'].append(data_dict_train['src'][i])
            sens_selected['trg'].append(data_dict_train['trg'][i])

    return sens_selected


def select_data(file_id, sent_id, user_answer, doc_source, N, data_dict_train, train_sen_vecs, src_vocab, word_embs):

    # Let's simply select closest document for now

    #For now, let's CHEAT and get the document
    data_dict_tune = {}
    data_dict_tune['src'] = []
    data_dict_tune['trg'] = []

    al_src = doc_source[sent_id]
    al_trg = user_answer.answer

    # Get tuning data by selecting similar sentences in the training data
    data_dict_tune = data_selection_emb(N, data_dict_train, train_sen_vecs, doc_source, src_vocab, word_embs)

    # Add the active learning sentences
    data_dict_tune['src'].append(al_src)
    data_dict_tune['trg'].append(al_trg)

    # FIXME: for now, just tune on the current sentence (debugging)
    #for sent in data_dict_train['src']:
    #    data_dict_tune['src'].append(sent)
    #for sent in data_dict_train['trg']:
    #    data_dict_tune['trg'].append(sent)


    return data_dict_tune


def online_adaptation(model, params, file_id, sent_id, user_answer, doc_source, current_hypothesis, data_dict_train, train_sen_vecs, src_vocab, word_embs):
    """
    Include here your code to adapt the model according
    to the human domain expert answer to the request

    Prototype of this function can be adapted to your need as it is internatl to this block
    """
    beat_logger.debug("### mt_lifelong_loop:online_adaptation")
    #print("### mt_lifelong_loop:online_adaptation")


    # Tune the model with data similar to the document we want to translate
    # FIXME: let's select 5k sentences for tuning
    N = 15000
    data_dict_tune = select_data(file_id, sent_id, user_answer, doc_source, N, data_dict_train, train_sen_vecs, src_vocab, word_embs)

    # prepare model for adaptation
    loop = prepare_model_for_tuning(model, params, data_dict_tune)

    # Run the adaptation
    new_model = loop()
    #NOTE: for debugging, just pass the previous model
    #new_model= model

    return new_model


class Algorithm:
    """
    Main class of the mt_lifelong_loop
    """
    def __init__(self):
        beat_logger.debug("### mt_lifelong_loop:init")
        self.model = None
        self.adapted_model = None
        self.data_dict_train = None
        self.data_dict_dev = None
        self.translate_params=TRANSLATE_DEFAULTS
        self.adapted_translate_params=TRANSLATE_DEFAULTS
        self.init_end_index = -1
        self.src_vocab = None
        self.train_sen_vecs = None
        self.data_dict_al = None
        self.data_dict_unsup = None

    def setup(self, parameters):
        #self.params={}
        self.translate_params['train']=TRAIN_DEFAULTS
        self.translate_params['model']={}

        self.translate_params['train']['seed']=int(parameters['seed'])
        self.translate_params['train']['model_type']=parameters['model_type']
        self.translate_params['train']['patience']=int(parameters['patience'])
        self.translate_params['train']['max_epochs']=int(parameters['max_epochs'])
        #self.translate_params['train']['eval_freq']=int(parameters['eval_freq'])
        # NOTE: Disable validation during finetuning
        self.translate_params['train']['eval_freq']=-1
        #self.translate_params['train']['eval_metrics']=parameters['eval_metrics']
        #self.translate_params['train']['eval_metrics']=None
        #self.translate_params['train']['eval_filters']=parameters['eval_filters']
        #self.translate_params['train']['eval_filters']=None
        #self.translate_params['train']['eval_beam']=int(parameters['eval_beam'])
        #self.translate_params['train']['eval_beam']=None
        #self.translate_params['train']['eval_batch_size']=int(parameters['eval_batch_size'])
        #self.translate_params['train']['eval_batch_size']=None
        self.translate_params['train']['save_best_metrics']=parameters['save_best_metrics']
        self.translate_params['train']['eval_max_len']=int(parameters['eval_max_len'])
        self.translate_params['train']['checkpoint_freq']=int(parameters['checkpoint_freq'])
        #self.translate_params['train']['n_checkpoints']=parameters['n_checkpoints']
        self.translate_params['train']['l2_reg']=int(parameters['l2_reg'])
        self.translate_params['train']['lr_decay']=parameters['lr_decay']
        self.translate_params['train']['lr_decay_revert']=parameters['lr_decay_revert']
        self.translate_params['train']['lr_decay_factor']=parameters['lr_decay_factor']
        self.translate_params['train']['lr_decay_patience']=int(parameters['lr_decay_patience'])
        self.translate_params['train']['gclip']=int(parameters['gclip'])
        self.translate_params['train']['optimizer']=parameters['optimizer']
        self.translate_params['train']['lr']=parameters['lr']
        self.translate_params['train']['batch_size']=int(parameters['batch_size'])
        self.translate_params['train']['save_optim_state']=False

        self.translate_params['train']['save_path']=Path("/not/used/because/beat_platform")
        #self.translate_params['train']['tensorboard_dir']="/lium/users/barrault/llmt/tensorboard"

        self.translate_params['model']['att_type']=parameters['att_type']
        self.translate_params['model']['att_bottleneck']=parameters['att_bottleneck']
        self.translate_params['model']['enc_dim']=int(parameters['enc_dim'])
        self.translate_params['model']['dec_dim']=int(parameters['dec_dim'])
        self.translate_params['model']['emb_dim']=int(parameters['emb_dim'])
        self.translate_params['model']['dropout_emb']=parameters['dropout_emb']
        self.translate_params['model']['dropout_ctx']=parameters['dropout_ctx']
        self.translate_params['model']['dropout_out']=parameters['dropout_out']
        self.translate_params['model']['n_encoders']=int(parameters['n_encoders'])
        self.translate_params['model']['tied_emb']=parameters['tied_emb']
        self.translate_params['model']['dec_init']=parameters['dec_init']
        self.translate_params['model']['bucket_by']="src"
        if parameters['max_len']=="None":
            self.translate_params['model']['max_len']=None
        else:
            self.translate_params['model']['max_len']=int(parameters['max_len'])
        self.translate_params['model']['direction']="src:Text -> trg:Text"

        self.adapted_translate_params = copy.deepcopy(self.translate_params)

        self.qe_model = None
        if parameters['direction'][11:13]=='de':
            # load EN-DE model
            kiwi_path = os.path.abspath(os.path.join(os.pardir,'openkiwi/trained_models/estimator_en_de.torch/estimator_en_de.torch'))
            self.qe_model = kiwi.load_model(kiwi_path)

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

        # recreate the translation model and train/dev data
        if self.model is None or self.data_dict_train is None:
            # Get the model after initial training
            dl = data_loaders[0]
            (data, _,end_index) = dl[0]

            # Store the baseline model
            if self.model is None:
                model_data = data['model'].value
                self.model = struct.pack('{}B'.format(len(model_data)), *list(model_data))

            if self.data_dict_train is None:
                data_dict = pickle.loads(data["processor_train_data"].text.encode("latin1"))
                self.data_dict_train, self.data_dict_dev = beat_separate_train_valid(data_dict)


        # Create a baseline Translator object from nmtpy
        self.translate_params['models'] = [self.model]
        self.translate_params['source'] = source
        translator = Translator(beat_platform=True, **self.translate_params)

        # train sentence vectors for data selection
        if self.train_sen_vecs is None:
            #Get the vocab from the opts of the model
            self.src_vocab = json.loads(translator.instances[0].opts['vocabulary']['src'])
            #Get the embeddings from the model's weights
            self.word_embs = translator.instances[0].enc.emb.weight
            # Create the sentence embeddings for the training data
            self.train_sen_vecs = get_sen_vecs(self.data_dict_train, self.src_vocab, self.word_embs)


        # Access incoming file information
        # See documentation for a detailed description of the mt_file_info
        file_info = inputs["processor_lifelong_file_info"].data
        file_id = file_info.file_id
        supervision = file_info.supervision
        time_stamp = file_info.time_stamp

        path_llnmt = '/home/barrault/msc/lifelongmt/'
        for p in ('original', 'adapted'):
            if not os.path.exists(path_llnmt + '{}'.format(p)):
                os.mkdir(path_llnmt+ '{}'.format(p))
        
        original_file = path_llnmt + 'original/{}'.format(file_id)
        adapted_file = path_llnmt + 'adapted/{}'.format(file_id)

        beat_logger.debug("mt_lifelong_loop::process: received document {} ({} sentences) to translate ".format(file_id, len(source)))
        #beat_logger.debug('mt_lifelong_loop::process: source = {}'.format(source))

        #TODO: prepare train/valid data for fine-tuning (eventually) -- might not be needed actually as the data is already contained in the training params -> this can be huge!
        current_hypothesis = run_translation(translator, source, file_id)
        with open(original_file, 'w') as f:
            for s in current_hypothesis:
                f.write(s)
                f.write('\n')
        # If human assisted learning is ON
        ###################################################################################################
        # Interact with the human if necessary
        # This section exchange information with the user simulation and ends up with a new hypothesis
        ###################################################################################################
        human_assisted_learning = supervision in ["active", "interactive"]
        # code not used!!
        if not human_assisted_learning:
            # In this method, see how to access initial training data to adapt the model
            # for the new incoming data
            self.adapted_model, adapted_translator = unsupervised_model_adaptation(source,
                                         file_id,
                                         self.train_data,
                                         self.model,
                                         translator,
                                         current_hypothesis
                                         )
            # update current_hypothesis with current model
            current_hypothesis = run_translation(adapted_translator, source, file_id)

        # If human assisted learning mode is on (active or interactive learning)
        while human_assisted_learning:

            # Create an empty request that is used to initiate interactive learning
            # For the case of active learning, this request is overwritten by your system itself
            # For now, only requests of type 'reference' are allowed (i.e. give me the reference translation for sentence 'sentence_id' of file 'file_id')

            if supervision == "active":
                # The system can send a question to the human in the loop
                # by using an object of type request
                # The request is the question asked to the system
                request = generate_system_request_to_user(file_id, source, current_hypothesis, self.qe_model)

                # Send the request to the user and wait for the answer
                message_to_user = {
                    "file_id": file_id,  # ID of the file the question is related to
                    "hypothesis": current_hypothesis[request['sentence_id']] ,  # The current hypothesis
                    "system_request": request,  # the question for the human in the loop
                }
                #beat_logger.debug("mt_lifelong_loop::process: send message to user: request={}".format(request))
                human_assisted_learning, user_answer = loop_channel.validate(message_to_user)

                # Take into account the user answer to generate a new hypothesis and possibly update the model
                adapted_model_data = online_adaptation(self.model, self.translate_params, file_id, request['sentence_id'], user_answer, source, current_hypothesis, self.data_dict_train, self.train_sen_vecs, self.src_vocab, self.word_embs)

                # Update the translator object with the current model
                self.adapted_model = struct.pack('{}B'.format(len(adapted_model_data)), *list(adapted_model_data))
                self.adapted_translate_params['models'] = [self.adapted_model]
                self.adapted_translate_params['source'] = source
                adapted_translator = Translator(beat_platform=True, **self.adapted_translate_params)

                # Generate a new translation
                new_hypothesis = run_translation(adapted_translator, source, file_id)
                # NOTE: let's debug by simply using the previous translation
                #new_hypothesis = current_hypothesis

                with open(adapted_file, 'w') as fad:
                    for s in new_hypothesis:
                        fad.write(s)
                        fad.write('\n')

                #beat_logger.debug("BEFORE online_adaptation: {}".format(current_hypothesis))
                #beat_logger.debug("AFTER online_adaptation : {}".format(new_hypothesis))
                # Update the current hypothesis with the new one
                current_hypothesis = new_hypothesis
                human_assisted_learning = False

            else:
                human_assisted_learning = False

        # End of human assisted learning
        # Send the current hypothesis
        #    self.init_end_index = 0
        #beat_logger.debug("HYPOTHESIS: {}".format(current_hypothesis))
        print("mt_lifelong_loop::process: FINISHED translated document {}: ".format(file_id))
        outputs["hypothesis"].write(mt_to_allies(current_hypothesis))

        if not inputs.hasMoreData():
            pass
        # always return True, it signals BEAT to continue processing
        return True



