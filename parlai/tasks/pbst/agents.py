import copy
import json
import os
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build


def raw_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'blended_skill_talk', dt + '.json')


def _processed_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'blended_skill_talk', dt + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        build(opt)
        # # get datafile
        # opt['parlaidialogteacher_datafile'] = _path(opt, '')

        super().__init__(opt, shared)