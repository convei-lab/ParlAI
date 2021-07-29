# #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


import os
import sys
from parlai.tasks.pbst.agents import parsed_data_path, prompt_candi_data_path, prompt_query_data_path, response_candi_data_path
import re
import json
import pandas as pd
import numpy as np
import copy
import random
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from typing import Tuple, Dict, List

from math import isclose
from collections import OrderedDict

from tqdm import tqdm
from icecream import ic
from parlai.scripts.eval_model import eval_model

RESOURCES = {
    'convai2': DownloadableFile(
        'http://parl.ai/downloads/convai2/convai2_fix_723.tgz',
        'convai2_fix_723.tgz',
        'd0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d',
    ), 
    'wizard_of_wikipedia': DownloadableFile(
        'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz',
        'wizard_of_wikipedia.tgz',
        '2a549627a83fea745efa2076a41d1c0078ad002ab2b54eae6a4e3d3d66ae24b7',
    ), 
    'empatheticdialogues': DownloadableFile(
        'http://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz',
        'empatheticdialogues.tar.gz',
        '56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d',
    )
}

SPLIT_RATIO = OrderedDict({'train': 0.6, 'valid': 0.3, 'test': 0.1})

def _parser_switch():
    parser_switch = {
        'convai2': {
            'files': ['train_both_original_no_cands.txt'], # 'valid_both_original_no_cands.txt'
            'func': _convai_parser,
        },
        'wizard_of_wikipedia': {
            'files': ['train.json', 'valid_random_split.json'], # 'test_random_split.json'
            'func': _wizard_of_wikipedia_parser,
        },
        'empatheticdialogues': {
            'files': ['train.csv', 'valid.csv'], # 'test.csv'
            'func': _empatheticdialogues_parser,
        }
    }
    return parser_switch


def _convai_parser(filepath
    ) -> Tuple[List[List[Dict, Dict]], List[List], List[Dict]]:
    debug = True

    def anonimize(p: str) -> str:
        return p.replace("partner's persona: ", "").replace("your persona: ", "")
    
    def swap(persona0, persona1):
        p0 = copy.deepcopy(persona0)
        return persona1, p0
    
    print('Parsing ConvAI2 on', filepath)
    prompts, dialogs, supports = [], [], [] # prompts are initial context pairs
    
    # Pre-processing
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        lines[i] = re.sub(r"^[0-9]+", "", line).strip()
    endline = len(lines)
        
    
    # Episode-level Collecting
    persona0, persona1, dialog, support = [], [], []
    
    for i, line in enumerate(lines):
        if line.startswith("partner's persona: "):
            persona0.append(anonimize(line))
        elif line.startswith("your persona: "):
            persona1.append(anonimize(line))
        else:
            utt_pair = line.split('\t')
            assert len(utt_pair) == 2
            if utt_pair[0] == '__SILENCE__':
                dialog.append(utt_pair[1])
                persona0, persona1 = swap(persona0, persona1)
            else:
                dialog.extend(utt_pair)
            if i < endline and lines[i+1].startswith("partner's persona: "):
                prompts.append([{"partner's persona: ": persona0}, {"your persona: : ", persona1}])
                dialogs.append([u.strip() for u in dialog])
                supports.append({})
                persona0, persona1 = [], []
    
    assert len(prompts) == len(dialogs) == len(supports)

    if debug:
        for i, (prompt, dialog, support) in enumerate(zip(prompts, dialogs, supports)):
            if i == 5:
                break
            print('\n'.join(prompt[0]))
            print('\n'.join(prompt[1]))
            print('\n'.join([f'{utt}\n\t{"\n\t".join(sup)})' for utt, sup in zip(dialog, support)]))
            input()

    return prompts, dialogs, supports

def _wizard_of_wikipedia_parser(filepath
    ) -> Tuple[List[List[Dict, Dict]], List[List], List[Dict]]:

    debug = True

    print('Parsing wizard_of_wikipedia on', filepath)
    prompts, dialogs, supports = [], [], []

    with open(filepath, 'r') as file:
        wow = json.load(file)
    for i, episode in enumerate(wow):
        # keys: 1 'chosen_topic', 1 'chosen_topic_passage (title?)', X 'persona', 1 'wizard_eval', X 'dialog'
        topic = episode['chosen_topic']
        persona = episode['persona']
        wizard_eval = episode['wizard_eval']
        dialog = episode['dialog']
        chosen_topic_passage = episode['chosen_topic_passage']

        if debug and i < 2:
            print('topic', topic)
            print('persona', persona)
            print('wizard_eval', wizard_eval)
            print('chosen_topic_passage', chosen_topic_passage)
            print('len passage', len(chosen_topic_passage))
            print('len diag', len(dialog))
            print('utts', [utt['text'] for utt in dialog])
            input()
            
            for j, utt_dic in enumerate(dialog):
                # each utt dic in dialog has keys: 1 speaker, 1 text, 1 checked_passage (title), 1 checked_sentence (article), 7 or 10 retrieved_passages
                if j < 3:
                    break
                speaker = utt_dic['speaker']
                text = utt_dic['text']
                print('\tspeaker', speaker)
                print('\ttext', text)
                if 'checked_sentence' in utt_dic and 'checked_passage' in utt_dic: # guiding only
                    checked_passage = utt_dic['checked_passage']
                    checked_sentence = utt_dic['checked_sentence']
                    print('\tchecked_sentence', checked_sentence)
                    print('\tchecked_passage', checked_passage)
                if 'candidate_responses' in utt_dic: # guided only
                    retrieved_topics = utt_dic['retrieved_topics'] # 7 topics same above
                    retrieved_passages = utt_dic['retrieved_passages'] # 7 topics as key with its passage as value
                    print('\tretrieved_topics', retrieved_topics)
                    print('\tretrieved_passages', retrieved_passages)
                    candidate_responses = utt_dic['candidate_responses'] # 100 utterance sentences
                    # print('\tcandidate_responses', candidate_responses)
                print()
                input()
                
        if dialog[0]['speaker'].endswith('Apprentice'): 
            # Apprentice first and Wizard second
            passage = chosen_topic_passage[0]
            checked_passage = [utt_dic['checked_passage'] for utt_dic in dialog]
            checked_sentence = [utt_dic['checked_sentence'] for utt_dic in dialog]
            dialogs.append([utt['text'].strip() for utt in dialog])
        else: 
            # Wizard first (dropped) and Apprentice second
            passage = chosen_topic_passage[1]
            dialogs.append([utt['text'].strip() for utt in dialog[1:]])
            checked_passage = [utt_dic['checked_passage'] for utt_dic in dialog[1:]]
            checked_sentence = [utt_dic['checked_sentence'] for utt_dic in dialog[1:]]
        prompts.append([{'topic': topic}, {'topic': topic, 'passage': passage}])
        supports.append({'checked_passage': checked_sentence, 'checked_sentence': checked_sentence})
        
    assert len(prompts) == len(dialogs) == len(supports)

    if debug:
        for i, (prompt, dialog, support) in enumerate(zip(prompts, dialogs, supports)):
            if i == 5:
                break
            print('\n'.join(prompt[0]))
            print('\n'.join(prompt[1]))
            print('\n'.join([f'{utt}\n\t{"\n\t".join(sup)})' for utt, sup in zip(dialog, support)]))
            input()

    return prompts, dialogs, supports

def _empatheticdialogues_parser(filepath
    ) -> Tuple[List[List[Dict, Dict]], List[List], List[Dict]]:
    debug = True

    print('Parsing empatheticdialogues on', filepath)
    prompts, dialogs, supports = [], [], []

    situations, emotions = [], []

    # Preprocessing
    df = pd.read_csv(filepath, usecols=range(8), sep=',', lineterminator='\n', quotechar="`")
    df['prompt'] = df['prompt'].str.replace('_comma_', ',')
    df['utterance'] = df['utterance'].str.replace('_comma_', ',')
    
    # Collecting
    situations = df.groupby('conv_id').agg({'prompt':lambda x: list(x)[0]})['prompt']
    emotions = df.groupby('conv_id').agg({'context':lambda x: list(x)[0]})['context']
    dialogs = df.groupby('conv_id').agg({'utterance':lambda x:list(x)})['utterances']

    # Check
    remove_idx = []
    for i, dialog in enumerate(dialogs):
        # Some episodes have only one turn! So we drop the such rows, 
        # as we cannot get a pair of initial utterances
        if len(dialog) != 2: 
            remove_idx.append(i)
    situations = situations.drop(situations.index[remove_idx]).tolist()
    emotions = emotions.drop(emotions.index[remove_idx]).tolist()
    dialogs = dialogs.drop(dialogs.index[remove_idx]).tolist()
    dialogs = [list(map(str.strip, dialog)) for dialog in dialogs]

    # Added extra information of emotion labels to the leader. Fair enough?
    for situation, emotion in zip(situations, emotions):
        prompts.append([{'situation': situation, 'emotion': emotion}, {}])
        supports.append({})
    
    assert len(prompts) == len(dialogs) == len(supports)

    if debug:
        for i, (prompt, dialog, support) in enumerate(zip(prompts, dialogs, supports)):
            if i == 5:
                break
            print('\n'.join(prompt[0]))
            print('\n'.join(prompt[1]))
            print('\n'.join([f'{utt}\n\t{"\n\t".join(sup)})' for utt, sup in zip(dialog, support)]))
            input()

    return prompts, dialogs, supports

def parse_subtask_dataset(opt: str, subtask: str
    ) -> Tuple[List[List[str, str]], List[List[str]], List[List[str]]]: 
    # collect the propmts and seed utterance pair from entire datatset
    prompts, dialogs, supports = [], [], []

    # identify correct parser and iterate all files to parse
    parser = _parser_switch()[subtask]
    for file in parser['files']:
        filepath = os.path.join(opt['datapath'], 'pbst', subtask, file)
        prompt, dialog, support = parser['func'](filepath)
        prompts.extend(prompt)
        dialogs.extend(dialog)
        supports.extend(support)
    return prompts, dialogs, supports

def write_subtask_parses(opt, subtask, prompts, dialogs, supports):
    
    parsed_path = parsed_data_path(opt, subtask)
    parses = list(zip(prompts, dialogs, supports))
    with open(parsed_path, 'w') as f:
        json.dump(parses, f)
    print(f'{len(prompts)} parsing results from {subtask} saved to {parsed_path}')
    
def write_prompt_retrieval_queries(opt, subtask, retri_query):
    retri_query_path = prompt_query_data_path(opt, subtask)
    with open(retri_query_path, 'w') as f:
        json.dump(retri_query, f)
    print(f'{len(retri_query)} retrieval queires for {subtask} saved to {retri_query_path}')
    
def write_prompt_retrieval_candidates(opt, subtask, retri_candi):
    retri_candi = np.unique(np.array(retri_candi))
    retri_candi_path = prompt_candi_data_path(opt, subtask)
    with open(retri_candi_path, 'w') as f:
        json.dump(retri_candi, f)
    print(f'{len(retri_candi)} retrieval candidates for {subtask} saved to {retri_candi_path}')
    
def build_prompt_retrieval_dataset(opt, subtask, dialogs, prompts):    
    # build query-to-candidate dataset
    
    # queries, candidates;
    write_prompt_retrieval_candidates(opt, subtask, retri_candi)

    # file structure: utterance, label, candidates(splitted by '\t')
    trainset = []
    validset = []
    testset = []
    with open(seed_utterance_wow_path) as json_file:
        seed_utterance_wow = json.load(json_file)

    # only topic (cause we can get passage by retrieving)
    with open(l_wow_path) as json_file:
        l_wow = json.load(json_file)
        
    with open(f_wow_path) as json_file:
        f_wow = json.load(json_file)

    l_wow_unique = list(set(l_wow))
    f_wow_unique = list(set(f_wow))

    for i in tqdm(range(len(seed_utterance_wow))):
        dialog_dict_l = {}
        dialog_dict_l['text'] = seed_utterance_wow[i][0]
        dialog_dict_l['labels'] = l_wow[i]
        # dialog_dict_l['label_candidates'] = l_wow_unique
        
        dialog_dict_f = {}
        dialog_dict_f['text'] = seed_utterance_wow[i][1]
        dialog_dict_f['labels'] = f_wow[i]
        # dialog_dict_f['label_candidates'] = ㄹ_wow_unique

        if i <= int(len(seed_utterance_wow) * 0.3):
            # topic_inference_train_list.append(dialog_dict_l)
            topic_inference_train_list.append(dialog_dict_f)
        elif i > int(len(seed_utterance_wow) * 0.8) and i < int(len(seed_utterance_wow) * 0.9):
            # topic_inference_valid_list.append(dialog_dict_l)
            topic_inference_valid_list.append(dialog_dict_f)
        elif i > int(len(seed_utterance_wow) * 0.9):
            # topic_inference_test_list.append(dialog_dict_l)
            topic_inference_test_list.append(dialog_dict_f)

    f = open(dpath + '/fixed_candidates.txt', 'w')
    for candidate in f_wow_unique:
        f.write(txt_escape(candidate) + '\n')
    print("Saved candidate file at", dpath + '/fixed_candidates.txt')       
    f.close()      
    
    with open(dpath + '/train.json', "w") as json_file:
        json.dump(topic_inference_train_list, json_file)
    print("Saved file at", dpath + '/train.json')

    # Due to storge issue, use train file as valid/test file
    with open(dpath + '/valid.json', "w") as json_file:
        json.dump(topic_inference_valid_list, json_file)
    print("Saved file at", dpath + '/valid.json')

    with open(dpath + '/test.json', "w") as json_file:
        json.dump(topic_inference_test_list, json_file)
    print("Saved file at", dpath + '/test.json')        
    
def write_response_candidates(opt, response_candidates):
    # writing the maximum pool of response candidate sampled from each task datasets
    respo_candi_path = response_candi_data_path(opt)
    with open(respo_candi_path, 'w') as f:
        json.dump(response_candidates, f)
    print(f'{len(response_candidates)} response candidates from {opt["subtask"]} saved to {respo_candi_path}')
    
def align_opener(opt: str):
    # prompts are different: for teacher and model 
    subtasks = opt['subtasks']
    prmpdic, diagdic, sprtdic = {}, {}, {} # seed pairs are concatenated into a sentence
    respo_candi = []

    for subtask in subtasks:
        # task-wise parsing 
        prompts, dialogs, supports = parse_subtask_dataset(subtask)
        write_subtask_parses(opt, subtask, prompts, dialogs, supports)
        
        # convert into value matrix
        prmpdic[subtask] = np.array([[p1.values(), p2.values()] for p1, p2 in prompts])
        diagdic[subtask] = np.array(dialogs) # odds teacher, evens model
        sprtdic[subtask] = np.array([s.values() for s in supports])
        
        # benchmark response candidates
        respo_candi.append(dialogs)
        
    # benchmark response candidates
    respo_candi = np.unique(np.array(respo_candi))
    response_candi_data_path(opt, respo_candi)

    # in-domain utterance-to-prompt alignment
    for subtask in subtasks:
        build_prompt_retrieval_dataset(opt, subtask, diagdic[subtask], prmpdic[subtask])
    utt2prmp = {}
    for orgtask in subtasks:
        utt2prmp[orgtask] = {}
        for trgtask in subtasks:
            if orgtask == trgtask:
                utt2prmp[subtask] = (seeds[:,0] + seeds[:,1]).tolist()
                seed2ctx[orgtask][trgtask] = ctxdic[orgtask]
    
    # cross-domain utterance-to-prompt alignment
    for trgtask in subtasks:
        if orgtask != trgtask:
            # align seeds with inter-task contexts
            seed2ctx[orgtask][trgtask] = ctxalign(orgtask, trgtask, seeddic[orgtask], ctxdic[trgtask], 'lexical_retrieval')
    
    context = []

    for srctaskid, srctask in enumerate(subtasks):

        context_length = len(lcm[srctaskid][0])
        alignedseed = seed_dic[srctask]
        
        for context_id in range(context_length):
            episode = {}
            episode['source_task'] = srctask
            episode['leader'] = {}
            episode['follower'] = {}

            episode['leader']['seed'] = alignedseed[context_id][0]
            episode['follower']['seed'] = alignedseed[context_id][1]
            episode['leader']['context'] = {}
            episode['follower']['context'] = {}

            for alignedtaskid, alignedtask in enumerate(subtasks):
                episode['leader']['context'][alignedtask] = lcm[srctaskid][alignedtaskid][context_id]

            for alignedtaskid, alignedtask in enumerate(subtasks):
                episode['follower']['context'][alignedtask] = fcm[srctaskid][alignedtaskid][context_id]

            context.append(episode)

 
    return context, response_candidates

def ctxalign(seed_queries, contextual_docs, teacher, origin, target, subtaskpath):
    '''
        retreival: query (utterance) -> document (a set of context description)
    '''
    retrieved_doc = []

    # Semantic Retreival (e.g. poly-encoder, DPR)
    if teacher == 'semantic':
        parlai_data_path = subtaskpath[:subtaskpath.find('pbst')]

        opt = {}
        if target == 'convai2':
            opt['task'] = 'persona_inference:retrieval'
        elif target == 'wizard_of_wikipedia':
            opt['task'] = 'topic_inference:retrieval'
        elif target == 'empatheticdialogues':
            opt['task'] = 'emotion_inference:retrieval'
        else:
            raise RuntimeError('Unimplemented subtask')

        split = opt['task'].split(':')

        # --world-logs true --report-filename ~/bst/convai_generation.json
        # opt['model_file'] = '/home/minju/bst/models/' + split[0] + '/tmp/model'
        opt['model'] = 'transformer/biencoder'
        opt['eval_candidates'] = 'inline'
        opt['fixed_candidates_path'] = parlai_data_path + split[0] + '/fixed_candidates.txt'
        opt['batchsize'] = 256
        opt['datatype'] = 'retrieval'
        opt['world_logs'] = parlai_data_path + split[0] + '/retrieval_report.json'
        opt['report_filename'] = parlai_data_path + split[0] + '/retrieval_report.json'
        opt['log_keep_fields'] = 'all'
        opt['num_examples'] = -1
        opt['display_examples'] = False
        opt['save_format'] = 'conversations'

        eval_list = []

        candidates_path = parlai_data_path + split[0] + '/fixed_candidates.txt'
        f = open(candidates_path, 'r')
        candidates = f.readlines()
        f.close()

        for query in seed_queries:
            # input_dict = {'text': query, 'label_candidates': candidates}
            input_dict = {'text': query, 'labels': candidates[0]}
            eval_list.append(input_dict)
            
        with open(parlai_data_path + split[0] + '/retrieval.json', "w") as json_file:
            json.dump(eval_list, json_file)
        print("Saved queries to", parlai_data_path + split[0] + '/retrieval.json')

        eval_model(opt)

        # Open retrieval result (jsonl file)
        retrieval_result_path = opt['report_filename'] + 'l'

        with open(retrieval_result_path, 'r') as json_file:
            json_list = list(json_file)
        retrieval_result = []
        for json_str in json_list:
            result = json.loads(json_str)
            retrieval_result.append(result)

        retrieved_doc = []
        for retrieved in retrieval_result:
            retrieved_doc.append(retrieved['dialog'][0][1]['text'])

    # Random Retrieval
    elif teacher == 'random':
        doc_ids = list(range(len(contextual_docs)))
        retrieved_doc_idx = random.choices(doc_ids, k=len(seed_queries))
        retrieved_doc = contextual_docs[retrieved_doc_idx]

    # TODO Manual Retrieval (e.g. BST -> 이 경우 context가 좀 더 단순해져야 한다. 현재 leading/following 불필요)

    # Lexical Retrieval??
    elif teacher == 'lexical_retrieval':

        parlai_data_path = subtaskpath[:subtaskpath.find('/pbst')]
        src2trg = f'{origin}->{target}'

        # Loading TF-IDF Retriever Model
        tr_opt = {}
        if target.startswith('convai2'):
            task = 'persona_inference'
        elif target.startswith('wizard_of_wikipedia'):
            task = 'topic_inference'
        elif target.startswith('empatheticdialogues'):
            task = 'emotion_inference'
        else:
            raise RuntimeError('Unimplemented retrieval task')
        tr_opt['task'] = f'{task}:{teacher}'
        tr_opt['model'] = 'tfidf_retriever'
        tr_opt['model_file'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/model'
        tr_opt['eval_candidates'] = 'inline'
        tr_opt['fixed_candidates_path'] = None
        tr_opt['batchsize'] = 256
        tr_opt['datatype'] = f'{teacher}/{src2trg}/query'
        tr_opt['label_candidates_file'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/fixed_candidates.txt'
        tr_opt['world_logs'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/results.jsonl'
        tr_opt['report_filename'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/model_report.json'
        tr_opt['log_keep_fields'] = 'all'
        tr_opt['num_examples'] = -1
        tr_opt['display_examples'] = False
        tr_opt['save_format'] = 'conversations'
        
        # prepare candidate for the retrieval, which are contextual documents
        # TODO 민주는 fixed candidates을 leading following 다 뭉쳐놨다. 나는 분리할 줄 알았다. 이 둘을 구분하는 것은 안 중요한가?
        with open(tr_opt['label_candidates_file'], 'r') as f:
            candidates = f.readlines()
        
        # prepare queries of the retireval, which are the seed utterances
        eval_list = []
        for query in seed_queries:
            # input_dict = {'text': query, 'labels': candidates[0], 'label_candidates': candidates}
            input_dict = {'text': query, 'labels': candidates[0]}
            eval_list.append(input_dict)
        
        # save the queries as files
        retrieval_query_path = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/query.json'
        retrieval_dirpath = retrieval_query_path.rsplit('/', 1)[0]
        if os.path.exists(retrieval_dirpath):
            build_data.remove_dir(retrieval_dirpath)
        build_data.make_dir(retrieval_dirpath)
        with open(retrieval_query_path, "w+") as json_file:
            json.dump(eval_list, json_file)
        print("Saved queries to", retrieval_query_path)
 
        # run parali retreival task, which then saves the retrieval results in as a 'world_logs' file
        eval_model(tr_opt)

        # load retrieval result (as in jsonl file) to read and return the retrieved documents
        with open(tr_opt['world_logs'], 'r') as json_file:
            json_list = list(json_file)

        retrieval_result = []
        for json_str in json_list:
            result = json.loads(json_str)
            if 'text' not in result['dialog'][0][1]:
                result['dialog'][0][1]['text'] = ""
                # result['dialog'][0][1]['candidate_ids'] = []
                # result['dialog'][0][1]['text_candidates'] = [] 
                # result['dialog'][0][1]['candidate_scores'] = []
            retrieval_result.append(result)

        for retrieved in retrieval_result:
            retrieved_doc.append(retrieved['dialog'][0][1]['text'])

        print('*'*5, "Contextual Alignment Example", '*'*5)
        print("Query:", seed_queries[0])
        print("Retreived Document:", retrieved_doc[0])
        print()

    return retrieved_doc



def build_blended_prompt(opt):
    version = 'v0.0'
    dpath = os.path.join(opt['datapath'], 'pbst')
    
    if not build_data.built(dpath, version):
        subtaskpaths = []

        for subtask in opt['subtasks']:
            subtaskpath = os.path.join(dpath, subtask)
            subtaskpaths.append(subtaskpath)
        
        logging.info('building data: ' + dpath)
        if build_data.built(dpath):
            y = None
            while y in ['y', 'n']:
                y = input('An older version of exists. Removing all outdated files. This may remove the benchmark dataset you built. Proceeds? y or n')
            build_data.remove_dir(dpath) if y == 'y' else sys.exit(1)
        build_data.make_dir(dpath)

        # # Download the data.
        # for subtask, subtaskpath in zip(opt['subtasks'], subtaskpaths):
        #     build_data.make_dir(subtaskpath)
        #     downloadable_file = RESOURCES[subtask]
        #     downloadable_file.download_file(subtaskpath) 

        if 'empatheticdialogues' in opt['subtasks']:
            # Move empatheticdialogues to parent directory
            # (ED 데이터셋만 내부폴더가 하나 더 생긴다. tar.gz라서 그런듯.)
            from shutil import move
            ed_path = subtaskpaths[opt['subtasks'].index('empatheticdialogues')]
            srcdir = os.path.join(ed_path, 'empatheticdialogues')
            if os.path.isdir(srcdir):
                for filename in os.listdir(srcdir):
                    move(os.path.join(srcdir, filename), os.path.join(ed_path, filename))
                os.rmdir(os.path.join(ed_path, 'empatheticdialogues'))

        context, random_candidates = align_context_and_response(opt, subtaskpaths)
        
        blended_context_path = os.path.join(dpath, 'blended_context.jsonl')
        with open(blended_context_path, 'w') as fout:
            # json.dump(context, fout)
            for dic in context:
                json.dump(dic, fout) 
                fout.write("\n")

        context_splits = lined_data_split(context, dpath, SPLIT_RATIO, randomized=True)
        

        parlai_formatter(dpath, opt)

        # Mark the data as built.
        build_data.mark_done(dpath, version)


def lined_data_split(json_list, dpath, split_ratio: OrderedDict, randomized=True):
    # sr = split_ratio = OrderedDict({'train': 0.6, 'val': 0.3, 'test': 0.1})

    # filepath = '/home/theorist/data3/ParlAI/data/pbst/pbst.jsonl'
    # with open(filepath, 'r') as json_file:
    #     data = json_list = [json.loads(jline) for jline in json_file.read().splitlines()]

    data = json_list

    # TODO 버그 있을 것만 같은 코드
    ds = dataset_size = len(json_list)

    assert isclose(sum([v for v in split_ratio.values()]), 1) # escape overflow
    ss = splitset_size = {k: round(ds * v) for k, v in split_ratio.items()}

    # Random sampling
    if randomized:
        random.shuffle(data)

    def greedy_split(data, sample_ratio):
        split_index = int(round(len(data) * sample_ratio, 6)) # 6의 자리 반올림
        sampled_data = data[:split_index]
        remained_data = data[split_index:]
        return sampled_data, remained_data

    original_length = len(data)
    sd = split_data = OrderedDict({split_name: None for split_name in split_ratio.keys()})

    remained_ratio = 1
    for i, (split_name, split_ratio) in enumerate(split_ratio.items()):
        sample_ratio = split_ratio / remained_ratio
        sd[split_name], data = greedy_split(data, sample_ratio)
        remained_ratio -= split_ratio

    assert isclose(remained_ratio, 0, abs_tol=1e-5), "Errors in split ratio"
    split_lengths = [len(d) for d in sd.values()]
    assert sum(split_lengths) == original_length, "Some samples in datset is remained after split"

    for split_name, dataset in sd.items():
        bc = os.path.join(dpath, f'blended_context_{split_name}.jsonl')
        pbc = os.path.join(dpath, f'pretty_blended_context_{split_name}.jsonl')
        with open(bc, 'w') as outputfile, open(pbc, 'w') as prettyfile:
            for sample in dataset:
                assert isinstance(sample, Dict)
                json.dump(sample, outputfile)
                outputfile.write('\n')
                json.dump(sample, prettyfile, indent=4)
                prettyfile.write('\n')
    return split_data


def parlai_formatter(dpath: str, opt: List):
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """
    datatypes = ['train', 'valid', 'test']
    for datatype in datatypes:

        load_path = os.path.join(dpath, f'blended_context_{datatype}.jsonl')
        save_path = os.path.join(dpath, f'blended_context_{datatype}.txt')

        print(f'Loading {load_path}.')
        # with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
        #     data = json.load(f_read)
        data = []
        with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
            for line in f_read:
                data.append(json.loads(line))

        print(f'Saving to {save_path}')
        subtasks = opt['subtasks']
        with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
            for episode in data:
                num_entries = 1
                entry_idx = 0
                for entry_idx in range(num_entries):
                    line = paralai_liner(episode, num_entries, entry_idx, subtasks)
                    f_write.write(f'{line} \n')


def paralai_liner(episode: dict, num_entries: int, entry_idx: int, subtasks: List) -> str:
    """
    Return the line to print in the reformatted file.
    """
    episode_done = entry_idx == num_entries - 1

    if entry_idx == 0:
        leader_context = '\n'.join([f"{episode['leader']['context'][task]}" for task in subtasks])
        follower_context = '\n'.join([f"{episode['follower']['context'][task]}" for task in subtasks])
        context_dataset = f"context dataset: {episode['source_task']}"
        original_context = '\n'.join([leader_context, follower_context, context_dataset]) + '\n'

    else:
        original_context = ''
    input_utterance = episode['leader']['seed']
    model_label = episode['follower']['seed']

    # Compile into text string
    parts = {
        'text': input_utterance,
        'labels': model_label,
        'expertise0': episode['source_task'],
        'expertise1': episode['source_task']
    }
    assert all([isinstance(part, str) for part in parts.values()])
    line = '\t'.join([f'{key}:{respo_escape(value)}' for key, value in parts.items()])

    # Add episode_done
    if episode_done:
        line += '\tepisode_done:True'

    return line


def respo_escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
        
def retri_escape(value: str) -> str:
    return str(value).replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')