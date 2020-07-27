import copy
import h5py
import numpy
import pickle
import os
import torch
import platform
import logging

import nmtpytorch
from nmtpytorch import logger
from nmtpytorch import models
from nmtpytorch.mainloop import MainLoop
from nmtpytorch.config import Options, TRAIN_DEFAULTS
from nmtpytorch.utils.misc import setup_experiment, fix_seed
from nmtpytorch.utils.device import DeviceManager
from nmtpytorch.utils.beat import beat_separate_train_valid

from pathlib import Path

import ipdb

beat_logger = logging.getLogger('beat_lifelong_mt')

class Algorithm:
    # initialise fields to store cross-input data (e.g. machines, aggregations, etc.)
    def __init__(self):
        self.distrib_nb = 4
        self.iv_size = 50
        self.plda_rank = 10
        self.feature_size = None


    def setup(self, parameters):
        self.params={}
        self.params['train']=TRAIN_DEFAULTS
        self.params['model']={}

        self.params['train']['seed']=int(parameters['seed'])
        self.params['train']['model_type']=parameters['model_type']
        self.params['train']['patience']=int(parameters['patience'])
        self.params['train']['max_epochs']=int(parameters['max_epochs'])
        self.params['train']['eval_freq']=int(parameters['eval_freq'])
        self.params['train']['eval_metrics']=parameters['eval_metrics']
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
        #self.params['train']['tensorboard_dir']="/lium/users/barrault/llmt/tensorboard"

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


    # this will be called each time the sync'd input has more data available to be processed
    def process(self, data_loaders, outputs):

        dl = data_loaders[0]
        (data, _,end_data_index) = dl[0]

        # separate train and dev data
        data_dict = pickle.loads(data["train_data"].text.encode("latin1"))
        self.data_dict_train, self.data_dict_dev = beat_separate_train_valid(data_dict)


        self.params['data']={}
        self.params['data']['train_set']={}
        self.params['data']['train_set']['src']=self.data_dict_train['src']
        self.params['data']['train_set']['trg']=self.data_dict_train['trg']
        self.params['data']['val_set']={}
        self.params['data']['val_set']['src']=self.data_dict_dev['src']
        self.params['data']['val_set']['trg']=self.data_dict_dev['trg']
        self.params['vocabulary']={}
        self.params['vocabulary']['src']=data['source_vocabulary'].text
        self.params['vocabulary']['trg']=data['target_vocabulary'].text
        self.params['filename']='/not/needed/beat_platform'
        self.params['sections']=['train', 'model', 'data', 'vocabulary']
        opts = Options.from_dict(self.params,{})

        setup_experiment(opts, beat_platform=True)
        dev_mgr = DeviceManager("gpu")

        # If given, seed that; if not generate a random seed and print it
        if opts.train['seed'] > 0:
            seed = fix_seed(opts.train['seed'])
        else:
            opts.train['seed'] = fix_seed()

        # Instantiate the model object
        model = getattr(models, opts.train['model_type'])(opts=opts, beat_platform=True)

        beat_logger.info("Python {} -- torch {} with CUDA {} (on machine '{}')".format(
            platform.python_version(), torch.__version__,
            torch.version.cuda, platform.node()))
        beat_logger.info("nmtpytorch {}".format(nmtpytorch.__version__))
        beat_logger.info(dev_mgr)
        beat_logger.info("Seed for further reproducibility: {}".format(opts.train['seed']))

        loop = MainLoop(model, opts.train, dev_mgr, beat_platform = True)
        model = loop()

        # The model is Pickled with torch.save() and converted into a 1D-array of uint8
        # Pass the model to the next block
        outputs['model'].write({'value': model}, end_data_index)

        beat_logger.debug("############## End of mt_train_model ############")
        return True






