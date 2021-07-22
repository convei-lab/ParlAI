import copy
import json
import os
from parlai.core.opt import Opt
from parlai.core.teachers import (
    ParlAIDialogTeacher,
    create_task_agent_from_taskname,
    MultiTaskTeacher,
    FixedDialogTeacher,
    DialogTeacher,
)
from .build import build


def raw_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.jsonl')


def _processed_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.txt')

def _generated_data_path(opt: Opt) -> str:
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'machine_generated.txt')
class PBSTTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)
        super().__init__(opt, shared)

class SelfchatTeacher(PBSTTeacher):
    # Dummy class to add arguments for interactive world.
    pass

class SelfmixTeacher(PBSTTeacher):
    # Dummy class to add arguments for interactive world.
    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        # # get datafile
        opt['parlaidialogteacher_datafile'] = _processed_data_path(opt)


        super().__init__(opt, shared)

class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):

        build(opt)
        opt = copy.deepcopy(opt)
        # # get datafile
        opt['parlaidialogteacher_datafile'] = _generated_data_path(opt)
        super().__init__(opt, shared)


def create_agents(opt):
    if not opt.get('interactive_task', False):
        return create_task_agent_from_taskname(opt)
    else:
        # interactive task has no task agents (they are attached as user agents)
        return []