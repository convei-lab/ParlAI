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
from parlai.utils.io import PathManager

# def seed_data_path(opt: Opt) -> str:
#     build(opt)
#     dt = opt['datatype'].split(':')[0]
#     return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.txt')

# def context_path(opt: Opt) -> str:
#     build(opt)
#     dt = opt['datatype'].split(':')[0]
#     return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.jsonl')

def parsed_data_path(opt: str, subtask: str) -> str:
    return os.path.join(opt['datapath'], 'pbst', f'parsed_{subtask}.json')

def prompt_query_data_path(opt: str, subtask: str) -> str:
    return os.path.join(opt['datapath'], 'pbst', 'prompt_retrieval', subtask, 'retrieval_queries.json')

def prompt_candi_data_path(opt: str, subtask: str) -> str:
    return os.path.join(opt['datapath'], 'pbst', 'prompt_retrieval', subtask, 'retrieval_candidates.json')

def prompt_retrieval_data_path(opt: Opt, subtask: str) -> str:
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', 'prompt_retrieval', subtask, dt + '.json')
    
def response_candi_data_path(opt: str) -> str:
    return os.path.join(opt['datapath'], 'pbst', 'responses_candidates.txt')

def blended_seed_data_path(opt: Opt) -> str:
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.txt')

def blended_context_data_path(opt: Opt) -> str:
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'blended_context_{dt}.jsonl')


def benchmark_data_path(opt: Opt) -> str:
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'pbst', f'machine_generated_{dt}.txt')


        
class PBSTTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = blended_seed_data_path(opt)
        super().__init__(opt, shared)

class SelfchatTeacher(PBSTTeacher):
    # Dummy class to add arguments for interactive world.
    pass

class SelfmixTeacher(PBSTTeacher):
    def __init__(self, opt, shared=None):

        build(opt)
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = blended_seed_data_path(opt)
        opt['benchmark_datafile'] = opt['outfile'] if opt['outfile'] is not None else _generated_data_path(opt)
        super().__init__(opt, shared)
        
class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):

        build(opt)
        opt = copy.deepcopy(opt)
        # # get datafile
        opt['parlaidialogteacher_datafile'] = benchmark_data_path(opt)
        super().__init__(opt, shared)

class ContextRetrievalTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        opt['datafile'] = retrieval_data_path(opt)
        self.id = 'persona_inference'
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.persona_inference = json.load(data_file)
        for dialog in self.persona_inference:
            text = dialog['text']
            if 'labels' in dialog.keys() and 'label_candidates' in dialog.keys():
                labels = dialog['labels']
                label_candidates = dialog['label_candidates']
                yield {"text": text, "labels": labels, 'label_candidates': label_candidates}, True
            elif 'labels' in dialog.keys():
                labels = dialog['labels']
                yield {'text': text, 'labels': labels}, True
            else:
                yield {"text": text}, True