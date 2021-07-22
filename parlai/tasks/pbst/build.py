# #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


import os
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

def build(opt):
    version = 'v0.0'
    dpath = os.path.join(opt['datapath'], 'pbst')
    subtaskpaths = []

    for subtask in opt['subtasks']:
        subtaskpath = os.path.join(dpath, subtask)
        subtaskpaths.append(subtaskpath)

    if not build_data.built(dpath, version):
        logging.info('building data: ' + dpath)
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
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

        context = _build_contextual_document(opt, subtaskpaths)
        blended_context_path = os.path.join(dpath, 'blended_context.jsonl')
        with open(blended_context_path, 'w') as fout:
            # json.dump(context, fout)
            for dic in context:
                json.dump(dic, fout) 
                fout.write("\n")

        context_splits = _split(context, dpath, SPLIT_RATIO, randomized=True)
        

        _create_parlai_format(dpath, opt)

        # Mark the data as built.
        build_data.mark_done(dpath, version)



def _convai_parser(filepath
    ) -> Tuple[List[str], List[str], List[str]]:
    print('Parsing ConvAI2 on', filepath)
    leading_contexts, following_contexts, seed_list = [], [], []
    
    # Pre-processing
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        lines[i] = re.sub(r"^[0-9]+", "", line).strip()
    
    # Collecting
    persona1, persona2, seed_pair = [], [], None
    episode_done = False
    for i, line in enumerate(lines):
        # print('Line', i, line) # for debug
        if line.startswith("partner's persona: "):
            persona1.append(line)
        elif line.startswith("your persona: "):
            persona2.append(line)
            episode_done = False
        elif not episode_done:
            seed_pair = line.split('\t')
            assert len(seed_pair) == 2
            leading_contexts.append('\n'.join(persona1)) 
            following_contexts.append('\n'.join(persona2))
            seed_list.append(seed_pair)
            episode_done = True
            persona1, persona2, seed_pair = [], [], []
            # print(person1_list[-1]) # for debug
            # print(person2_list[-1])
            # print(seed_list[-1])
            # input()
        else:
            continue
    
    assert len(leading_contexts) == len(following_contexts) == len(seed_list)

    # # for debug
    # for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
    #     if i == 2:
    #         break
    #     print(leading_context)
    #     print(following_context)
    #     print(seed)
    #     input()

    return leading_contexts, following_contexts, seed_list

def _wizard_of_wikipedia_parser(filepath
    ) -> Tuple[List[str], List[str], List[str]]:
    print('Parsing wizard_of_wikipedia on', filepath)
    leading_contexts, following_contexts, seed_list = [], [], []

    topic_list, passage_list, persona_list, seed_list = [], [], [], []
    with open(filepath, 'r') as file:
        wow = json.load(file)
    for i, episode in enumerate(wow):
        # keys: 'chosen_topic', 'persona', 'wizard_eval', 'dialog', 'chosen_topic_passage'
        topic = episode['chosen_topic']
        persona = episode['persona']
        wizard_eval = episode['wizard_eval']
        dialog = episode['dialog']
        chosen_topic_passage = episode['chosen_topic_passage']

        # for debug
        # print('topic', topic)
        # print('persona', persona)
        # print('wizard_eval', wizard_eval)
        # print('chosen_topic_passage', chosen_topic_passage)
        # print('len passage', len(chosen_topic_passage))
        # print('len dialog', len(dialog))
        # TODO 모든 시간대에 사용한 지식 sentence를 전부 제공한다. (o) vs 너무 길기 때문에 처음만 제공한다. (x)
        passage = ' '.join(chosen_topic_passage) 
        # passage = chosen_topic_passage[0] 
        if dialog[0]['speaker'].endswith('Apprentice'): # 1_Apprentice first and 0_Wizard second
            seed = [utt_dic['text'] for utt_dic in dialog[:2]]
        else: # 0_Wizard first and 1_Apprentice second
            seed = [utt_dic['text'] for utt_dic in dialog[1:3]]
        topic_list.append(topic)
        passage_list.append(passage)
        persona_list.append(persona)
        seed_list.append(seed)

        # # for debug
        # for j, utt_dic in enumerate(dialog):
        #     # keys: 'speaker', 'text', 'candidate_responses', 'retrieved_passages', 'retrieved_topics'
        #     speaker = utt_dic['speaker']
        #     text = utt_dic['text']
        #     if 'checked_sentence' in utt_dic and 'checked_passage' in utt_dic: # guiding only
        #         checked_sentence = utt_dic['checked_sentence']
        #         checked_passage = utt_dic['checked_passage']
        #     if 'candidate_responses' in utt_dic: # guided only
        #         candidate_responses = utt_dic['candidate_responses'] # 100 utterance sentences
        #     retrieved_passages = utt_dic['retrieved_passages'] # 7 topics as key with its passage as value
        #     retrieved_topics = utt_dic['retrieved_topics'] # 7 topics same above
        #     if j < 3:
        #         print('Speaker', speaker, text)
        #         input()

    assert  len(topic_list) == len(passage_list) == len(seed_list) == len(persona_list)

    for topic, passage, persona in zip(topic_list, passage_list, persona_list):
        # leading_contexts.append(f'Topic: {topic}\nPersona: {persona}') # Persona 떼어냈다.
        leading_contexts.append(f'topic: {topic}') 
        following_contexts.append(f'topic: {topic}\nknowledge: {passage}')
    
    
    assert len(leading_contexts) == len(following_contexts) == len(seed_list)

    # # for debug
    # for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
    #     if i == 2:
    #         break
    #     print(leading_context)
    #     print(following_context)
    #     print(seed)
    #     input()

    return leading_contexts, following_contexts, seed_list

def _empatheticdialogues_parser(filepath
    ) -> Tuple[List[str], List[str], List[str]]:
    print('Parsing empatheticdialogues on', filepath)
    leading_contexts, following_contexts, seed_list = [], [], []

    situation_list, emotion_list, seed_list = [], [], []

    # Preprocessing
    df = pd.read_csv(filepath, usecols=range(8), sep=',', lineterminator='\n', quotechar="`")
    df['prompt'] = df['prompt'].str.replace('_comma_', ',')
    df['utterance'] = df['utterance'].str.replace('_comma_', ',')
    
    # Collecting
    situation_list = df.groupby('conv_id').agg({'prompt':lambda x: list(x)[0]})['prompt']
    emotion_list = df.groupby('conv_id').agg({'context':lambda x: list(x)[0]})['context']
    seed_list = df.groupby('conv_id').agg({'utterance':lambda x: list(x)[:2]})['utterance']

    # Check
    remove_idx = []
    for i, seed in enumerate(seed_list):
        if len(seed) != 2: # some episodes have only one turn!
            remove_idx.append(i)
    situation_list = situation_list.drop(situation_list.index[remove_idx]).tolist()
    emotion_list = emotion_list.drop(emotion_list.index[remove_idx]).tolist()
    seed_list = seed_list.drop(seed_list.index[remove_idx]).tolist()

    assert len(situation_list) == len(emotion_list) == len(seed_list)

    # TODO Added extra information of emotion labels to the leader. Is this fair enough?
    for situation, emotion in zip(situation_list, emotion_list):
        leading_contexts.append(f'situation: {situation}\nemotion: {emotion}')
        following_contexts.append(f'')
    assert len(leading_contexts) == len(following_contexts) == len(seed_list)

    # # for debug
    # for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
    #     if i == 2:
    #         break
    #     print(leading_context)
    #     print(following_context)
    #     print(seed)
    #     input()

    return leading_contexts, following_contexts, seed_list

def parser_switch():
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

def _parse_task_dataset(subtask, subtaskpath
    ) -> Tuple[List[str], List[str], List[List[str]]]:
    # Collect the context and the seed utterance pair of each episode
    leading_contextss, following_contextss, seeds = [], [], []

    # Identify correct parser and iterate all files to parse
    parser = parser_switch()[subtask]
    for file in parser['files']:
        filepath = os.path.join(subtaskpath, file)
        fleading_contextss, ffollowing_contextss, fseeds = parser['func'](filepath)
        leading_contextss.extend(fleading_contextss)
        following_contextss.extend(ffollowing_contextss)
        seeds.extend(fseeds)
    return leading_contextss, following_contextss, seeds

def _retrieve_contextual_document(seed_queries, contextual_docs, mode, target, subtaskpaths):
    # Semantic Retreival (e.g. poly-encoder, DPR)
    if mode == 'semantic':
        # EDITED BY MINJU
        parlai_data_path = subtaskpaths[0][:subtaskpaths[0].find('pbst')]

        opt = {}
        if target == 'convai2':
            opt['task'] = 'persona_inference:retrieval'
        elif target == 'wizard_of_wikipedia':
            opt['task'] = 'topic_inference:retrieval'
        else:
            opt['task'] = 'emotion_inference:retrieval'

        split = opt['task'].split(':')

        # --world-logs true --report-filename ~/bst/convai_generation.json
        opt['model_file'] = '/home/minju/bst/models/' + split[0] + '/tmp/model'
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

        candidates_path = '/home/minju/ParlAI/data/' + split[0] + '/fixed_candidates.txt'
        f = open(candidates_path, 'r')
        candidates = f.readlines()
        f.close()

        for query in seed_queries:
            # input_dict = {'text': query, 'label_candidates': candidates}
            input_dict = {'text': query}
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

        return retrieved_doc

    # Random Retrieval
    elif mode == 'random':
        doc_ids = list(range(len(contextual_docs)))
        retrieved_doc_idx = random.choices(doc_ids, k=len(seed_queries))

    # TODO Manual Retrieval (e.g. BST -> 이 경우 context가 좀 더 단순해져야 한다. 현재 leading/following 불필요)

    # Lexical Retrieval??
    elif mode == 'lexical':
        # EDITED BY MINJU
        parlai_data_path = subtaskpaths[0][:subtaskpaths[0].find('pbst')]

        opt = {}
        if target == 'convai2':
            opt['task'] = 'persona_inference:retrieval'
        elif target == 'wizard_of_wikipedia':
            opt['task'] = 'topic_inference:retrieval'
        else:
            opt['task'] = 'emotion_inference:retrieval'

        split = opt['task'].split(':')

        # parlai eval_model -m ir_baseline -t emotion_inference --world-logs /home/minju/data1/ParlAI/data/emotion_inference/eval_result.jsonl --batchsize 256 --label-candidates-file /home/minju/data1/ParlAI/data/emotion_inference/fixed_candidates.txt
        opt['model'] = 'ir_baseline'
        opt['model_file'] = None
        opt['eval_candidates'] = 'inline'
        opt['fixed_candidates_path'] = None
        opt['batchsize'] = 256
        opt['datatype'] = 'retrieval'
        opt['label_candidates_file'] = parlai_data_path + split[0] + '/fixed_candidates.txt'
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
            # input_dict = {'text': query, 'labels': candidates[0], 'label_candidates': candidates}
            input_dict = {'text': query}
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

        print("Contextual alignment example")
        print("query", seed_queries[0])
        print("document", retrieved_doc[0])

        return retrieved_doc

    return retrieved_doc_idx
        


def _build_contextual_document(opt, subtaskpaths):
    # contexts are different: for leading speaker and following speaker 
    subtasks, nsubtasks = opt['subtasks'], len(opt['subtasks'])
    leading_context_dic, following_context_dic, seed_dic = {}, {}, {} # seed pairs are concatenated into a sentence
    
    # Collect task-wise contexts and seeds
    for subtask, subtaskpath in zip(subtasks, subtaskpaths):
        leading_contexts, following_contexts, seeds = _parse_task_dataset(subtask, subtaskpath)
        lc = os.path.join(opt['datapath'], 'pbst', f'leading_{subtask}_kb.json')
        fc = os.path.join(opt['datapath'], 'pbst', f'following_{subtask}_kb.json')
        su = os.path.join(opt['datapath'], 'pbst', f'seed_utterance_pairs_{subtask}.json')
        with open(lc, 'w') as outputfile1, open(fc, 'w') as outputfile2, open(su, 'w') as outputfile3:
            json.dump(leading_contexts, outputfile1)
            json.dump(following_contexts, outputfile2)
            json.dump(seeds, outputfile3)
        assert len(leading_contexts) == len(following_contexts) == len(seeds)
        leading_context_dic[subtask] = leading_contexts
        following_context_dic[subtask] = following_contexts
        seed_dic[subtask] = seeds
        print(f'{len(seeds)} contexts and seed utterances pairs were parsed from the {subtask}\n')
    
    lcm = leading_contextual_matrix = [[None]*nsubtasks for _ in range(nsubtasks)]
    fcm = following_contextual_matrix = copy.deepcopy(lcm)

    # Align inter-task relationhip between seeds and contexts
    for i, origin in enumerate(subtasks):
        seed_pairs = np.array(seed_dic[origin])
        leading_seeds, following_seeds = seed_pairs[:,0], seed_pairs[:,1]
        for j, target in enumerate(subtasks):
            if i == j:
                lcm[i][j] = leading_context_dic[origin]
                fcm[i][j] = following_context_dic[origin]
            else:
                leading_contexts = leading_context_dic[target]
                following_contexts = following_context_dic[target]
                # Retrieve contextual document from different task
                leading_doc_ids = _retrieve_contextual_document(leading_seeds, leading_contexts, 'lexical', target, subtaskpaths)
                following_doc_ids = _retrieve_contextual_document(following_seeds, following_contexts, 'lexical', target, subtaskpaths)
 
                # Align the seed with all the other subtask's context
                if target == 'convai2':
                    lcm[i][j] = [leading_contexts[i] for i in leading_doc_ids] # no dependency leader-follower
                    fcm[i][j] = [following_contexts[i] for i in following_doc_ids]
                elif target == 'wizard_of_wikipedia':
                    lcm[i][j] = [leading_contexts[i] for i in following_doc_ids] # follower (wizard) based
                    fcm[i][j] = [following_contexts[i] for i in following_doc_ids]
                elif target == 'empatheticdialogues':
                    lcm[i][j] = [leading_contexts[i] for i in leading_doc_ids] # leader (situation) based
                    fcm[i][j] = [following_contexts[i] for i in leading_doc_ids]
    
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

    # for git merge        
    # # for debug
    # for i, episode in enumerate(context):
    #     print('Episode', i, '*'*40)
    #     print('1 Source task', episode['source_task'])
    #     print('2 Leader\'s context')
    #     leader = episode['leader']
    #     for task, context in leader['context'].items():
    #         print('2+ Task', task)
    #         print(context)
    #     print('3 Leader\'s seed')
    #     print(leader['seed'])
    #     print()
    #     print('4 Follower\'s context')
    #     follower = episode['follower']
    #     for task, context in follower['context'].items():
    #         print('4+ Task', task)
    #         print(context)
    #     print('5 Follower\'s seed')
    #     print(follower['seed'])
    #     input()
    
 
    return context

def _split(json_list, dpath, split_ratio: OrderedDict, randomized=True):
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


def _create_parlai_format(dpath: str, opt: List):
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
                    line = _get_line(
                        episode, num_entries, entry_idx, subtasks
                    )
                    f_write.write(f'{line} \n')


def _get_line(episode: dict, num_entries: int, entry_idx: int, subtasks: List) -> str:
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
    }
    assert all([isinstance(part, str) for part in parts.values()])
    line = '\t'.join([f'{key}:{_escape(value)}' for key, value in parts.items()])

    # Add episode_done
    if episode_done:
        line += '\tepisode_done:True'

    return line


def _escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
