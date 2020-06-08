# You may import any python packages that will be available in the environment you will run this database in
# Environments can change based on the experiment's settings
from beat.backend.python.database import View

import os
import numpy as np
import re
import logging
#import struct
from collections import namedtuple
import pdb

beat_logger = logging.getLogger('beat_lifelong_mt')
beat_logger.setLevel(logging.DEBUG)

def get_file_list(path):
    pairs = []
    #pattern = re.compile(r'[^-]*-(?P<date>[^-]*).*\.txt')
    pattern = re.compile(r'[^-]*-(?P<date>[^-]*)-*(?P<time1>[^-.]*)-*(?P<time2>[^-\.]*).*\.txt')
    for name in os.listdir(path):
        #beat_logger.debug("Found file: {}".format(name))
        #print("Found file: {}".format(name))
        date, t1, t2 = pattern.search(name).groups()
        if t1 or t2:
            date = "{}.{}{}".format(date, t1, t2)
        #beat_logger.debug("DATE = {}".format(date))
        #print("DATE = {}".format(date))
        name = name[:-4]
        pairs.append([date, name])
    pairs.sort(key = lambda p: p[0])
    return pairs

def hashrand(a):
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    a = a & 0xffffffff
    a = a / 0xffffffff
    return a

class View(View):

    def __init__(self):
        super().__init__()
        self.pair = ""
        self.source_lang = "src"
        self.target_lang = "trg"



    # build the data for your view
    # split the raw data into (homogenous) bits and return a keyed iterable
    # (something with `.keys()` available to it, like a dict)
    # the key names must be the same as the output names for sets that use this view
    #    root_folder: the path to the root folder of the database's files (not always applicable)
    #    parameters: parameters passed to the view, defined in the metadata
    def index(self, root_folder, parameters):
        #beat_logger.debug("### allies_mt_internal::index: start")
        self.source_lang = parameters["source_lang"]
        self.target_lang = parameters["target_lang"]
        self.pair = '{}-{}'.format(self.source_lang, self.target_lang)

        fl = get_file_list('{}/{}/{}'.format(root_folder, self.pair, self.source_lang))
        Entry = namedtuple(
            'Entry', ['root_folder', 'filename', 'file_info', 'source', 'target']
        )
        ents = []
        for idx, f in enumerate(fl):
            #print("FILE = ", f)
            #print("TRAIN START={} TRAIN_END={} DEV_START={} DEV_END={} ||| FILE_DATE={}".format(parameters['start_date'], parameters['end_date'], parameters['dev_start_date'], parameters['dev_end_date'], f[0]), end=' ')

            in_train = ('start_date' in parameters) and (f[0] >= parameters['start_date']) and ('end_date' in parameters) and (f[0] <  parameters['end_date'])
            in_dev   = ('dev_start_date' in parameters) and (f[0] >= parameters['dev_start_date']) and ('dev_end_date' in parameters) and (f[0] < parameters['dev_end_date'])

            # pass if file date not in train/dev time span
            if not in_train and not in_dev:
                continue
            llmode = 'none'

            if ('interactive_ratio' in parameters):
                if float(parameters['interactive_ratio']) > hashrand(idx):
                    llmode = 'interactive'
                else:
                    llmode = 'active'
            ent = Entry(root_folder, f[1], { 'file_id': f[1], 'supervision': llmode, 'time_stamp': f[0][0:13], 'in_dev': in_dev }, [f[0]], [f[0]])
            ents.append(ent)
            #beat_logger.info("allies-mt-internal::index: ent = {}".format(ent))
            #print("allies-mt-internal::index: ent = {}".format(ent))
        #print("allies-mt-internal: ENTS = ")
        #for entry in ents:
        #    print("ENT:{} -> {}".format(entry[2], entry[5]) )
        return ents

    # returns a value at a specific index in the iterable for this view
    #   output: the specific output value requested
    #   index: the current index of the iterable
    def get(self, output, index):
        #beat_logger.debug("### allies_mt_internal::get: start")
        # to get the current object referenced by the given index:
        #       obj = self.objs[index]
        # note that this object is a named tuple, with fields equivalent to your keys from
        # the objects returned from the index function
        obj = self.objs[index]
        if output == 'file_info':
            return obj.file_info
        elif output == 'source':
            with open("{}/{}/{}/{}.txt".format(obj.root_folder, "en-fr", "en", obj.filename, 'r')) as fileh:
                text = fileh.readlines()
            text = list(map(str.strip, text))
            #beat_logger.info(" -----------------------------------------------------------------------------------------------  NEW SOURCE: {}  #lines: {}".format(obj.filename, len(text)))
            return { 'text': text }
        elif output == 'target':
            with open("{}/{}/{}/{}.txt".format(obj.root_folder, "en-fr", "fr", obj.filename, 'r')) as fileh:
                text = fileh.readlines()
            text = list(map(str.strip, text))
            return { 'text': text }
