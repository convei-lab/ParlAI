import copy
import json
import os
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
from parlai.utils.io import PathManager

class TopicInferenceTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        dt = opt['datatype'].split(':')[0]
        opt['datafile'] = os.path.join(opt['datapath'], 'topic_inference', dt + '.json')
        self.id = 'topic_inference'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.topic_inference = json.load(data_file)
        for dialog in self.topic_inference:
            text = dialog['text']
            labels = dialog['labels']
            yield {"text": text, "labels": labels}, True

class DefaultTeacher(TopicInferenceTeacher):
    pass