#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful utilities for logging actions/observations in a world.
"""

from random import random
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.core.message import Message

import copy
from tqdm import tqdm

KEEP_ALL = 'all'


class WorldLogger:
    """
    Logs actions/observations in a world and saves in a given format.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group('World Logging')
        agent.add_argument(
            '--log-keep-fields',
            type=str,
            default=KEEP_ALL,
            help='Fields to keep when logging. Should be a comma separated list',
        )
        return parser

    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self._set_keep_fields(opt)

        self._current_episodes = {}
        self._logs = []

        self.reset()

    def _set_keep_fields(self, opt):
        self.keep_fields = opt['log_keep_fields'].split(',')
        self.keep_all = KEEP_ALL in self.keep_fields

    def reset(self):
        for _, ep in self._current_episodes.items():
            self._add_episode(ep)

        self._current_episodes = {}

    def reset_world(self, idx=0):
        if idx not in self._current_episodes:
            return
        self._add_episode(self._current_episodes[idx])
        self._current_episodes[idx] = []

    def _add_msgs(self, acts, idx=0):
        """
        Add messages from a `parley()` to the current episode of logs.

        :param acts: list of acts from a `.parley()` call
        """
        msgs = []
        for act in acts:
            # padding examples in the episode[0]
            if not isinstance(act, Message):
                act = Message(act)
            if act.is_padding():
                break
            if not self.keep_all:
                msg = {f: act[f] for f in self.keep_fields if f in act}
            else:
                msg = act
            msgs.append(msg)

        if len(msgs) == 0:
            return
        self._current_episodes.setdefault(idx, [])
        self._current_episodes[idx].append(msgs)

    def _add_episode(self, episode):
        """
        Add episode to the logs.
        """
        self._logs.append(episode)

    def _is_batch_world(self, world):
        return (
            isinstance(world, BatchWorld) or isinstance(world, DynamicBatchWorld)
        ) and len(world.worlds) > 1

    def _log_batch(self, world):
        batch_act = world.get_acts()
        parleys = zip(*batch_act)
        for i, parley in enumerate(parleys):
            # in dynamic batching, we only return `batchsize` acts, but the
            # 'dyn_batch_idx' key in the task act corresponds the episode index
            # in the buffer
            idx = parley[0]['dyn_batch_idx'] if 'dyn_batch_idx' in parley[0] else i
            self._add_msgs(parley, idx=idx)
            if world.worlds[idx].episode_done():
                self.reset_world(idx=idx)

    def log(self, world):
        """
        Log acts from a world.
        """
        # log batch world
        if self._is_batch_world(world):
            self._log_batch(world)
            return

        # log single world
        acts = world.get_acts()
        self._add_msgs(acts)
        if world.episode_done():
            # add episode to logs and clear examples
            self.reset_world()

    def convert_to_labeled_data(self, episode):
        out = []
        text_lst = []
        for parley in episode:
            first_act, second_act = parley
            if 'text' in first_act:
                text_lst.append(first_act['text'])
            if second_act.get('id') != 'context':
                label = second_act.get('text')
                out.append(
                    {
                        'id': first_act.get('id', ''),
                        'text': '\n'.join(text_lst),
                        'labels': [label],
                        'episode_done': False,
                    }
                )
                text_lst = []
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def write_parlai_format(self, outfile):
        logging.info(f'Saving log to {outfile} in ParlAI format')
        with PathManager.open(outfile, 'w') as fw:
            for episode in tqdm(self._logs):
                ep = self.convert_to_labeled_data(episode)
                for act in ep:
                    txt = msg_to_str(act)
                    fw.write(txt + '\n')
                fw.write('\n')

    def write_conversations_format(self, outfile, world):
        logging.info(f'Saving log to {outfile} in Conversations format')
        Conversations.save_conversations(
            self._logs,
            outfile,
            world.opt,
            self_chat=world.opt.get('selfchat_task', False),
        )

    def write(self, outfile, world, file_format='conversations', indent=4):
        if file_format == 'conversations':
            self.write_conversations_format(outfile, world)
        else:
            # ParlAI text format
            self.write_parlai_format(outfile)

    def get_logs(self):
        return self._logs


class DebateLogger(WorldLogger):

    def _add_msgs(self, acts, idx=0):
        """
        Add messages from a `parley()` to the current episode of logs.

        :param acts: list of acts from a `.parley()` call
        """
        msgs = []
        for act_pair in acts:
            msg_pair = []
            for act in act_pair:
                # padding examples in the episode[0]
                if not isinstance(act, Message):
                    act = Message(act)
                if act.is_padding():
                    break
                if not self.keep_all:
                    msg = {f: act[f] for f in self.keep_fields if f in act}
                else:
                    msg = act
                msg_pair.append(msg)
            msgs.append(msg_pair)

        if len(msgs) == 0:
            return
        self._current_episodes.setdefault(idx, [])
        self._current_episodes[idx].append(msgs)

    def convert_to_labeled_data(self, episode, subtasks):
        """
        Write one episode into series of parl-ai format lines
        """
        debug = False
        out = []
        
        # dialogue initiation
        partner_context = []
        subtask_context = {}
        seeded = False
        ntasks = len(subtasks)
        assert len(episode) % ntasks == 0
        
        for i, expertise in enumerate(episode):
            line = {'id': '', 'text': '', 'labels': '', 'episode_done': False,}
            text_lst = []

            for j, (subtask, parley) in enumerate(zip(subtasks, expertise)):
                first_act, second_act = parley
                if first_act['id'] == 'context' and second_act['id'] == 'context': # context
                    text_lst.append(second_act['text'])
                    partner_context.append(first_act['text'])
                    subtask_context[f'context1_{subtask}'] = first_act['text']
                    subtask_context[f'context2_{subtask}'] = second_act['text']
                elif first_act['id'] == 'seed' and second_act['id'] == 'seed': # seed
                    if not seeded: # Writing first seed
                        text_lst.append(first_act['text'])

                        line['id'] = 'context'
                        line['text'] = '\n'.join(text_lst)
                        line['labels'] = [second_act['text']]
                        line['partner_context'] = '\n'.join(partner_context) if partner_context else ''
                        for k, v in subtask_context.items(): line[k] = v
                        out.append(line)
                        if debug: print('*line*', line, '\n'); input()
                        text_lst = []
                        seeded = True
                else: # utterance
                    if first_act['decision'] == '1':
                        line['id'] = first_act['id']
                        text_lst.append(first_act['text'])
                        line['context1_dataset'] = subtasks[j]
                    if second_act['decision'] == '1':
                        line['labels'] = [second_act['text']]
                        line['context2_dataset'] = subtasks[j]
                    first_verdict = first_act['verdict'].split(',')
                    second_verdict = second_act['verdict'].split(',')
                    suggestions1 = [[t, str(round(v, 2))] for t, v in first_act['beam_texts']]
                    suggestions2 = [[t, str(round(v, 2))] for t, v in second_act['beam_texts']]
                    for k in range(len(first_verdict)):
                        line[f'suggestion1_{subtask}_{k}'] = ' '.join(suggestions1[k]) + f' (verdict: {first_verdict[k]})'
                        line[f'suggestion2_{subtask}_{k}'] = ' '.join(suggestions2[k]) + f' (verdict: {second_verdict[k]})' 
                if debug: print(first_act); print(second_act); input()
            # In case of utterances, we collect the best while iteration.
            if second_act['id'] != 'context' and second_act['id'] != 'seed':
                line['text'] = '\n'.join(text_lst)
                out.append(line)
                if debug: print('*line*', line, '\n'); input()
                text_lst = []
                # TODO add random sampling for response candidates
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def convert_to_labeled_data2(self, episode, subtasks):
        """
        Write one episode into series of parl-ai format lines
        """
        debug = False
        out = []
        
        # dialogue initiation
        partner_context = []
        subtask_context = {}
        seeded = False
        ntasks = len(subtasks)
        assert len(episode) % ntasks == 0
        
        for i, expertise in enumerate(episode):
            line = {'id': '', 'text': '', 'labels': '', 'episode_done': False,}
            text_lst = [] # TODO this needs to go away

            for j, (subtask, parley) in enumerate(zip(subtasks, expertise)):
                first_act, second_act = parley
                if first_act['id'] == 'context' and second_act['id'] == 'context': # context
                    text_lst.append(second_act['text'])
                    partner_context.append(first_act['text'])
                    subtask_context[f'context1_{subtask}'] = first_act['text']
                    subtask_context[f'context2_{subtask}'] = second_act['text']
                elif first_act['id'] == 'seed' and second_act['id'] == 'seed': # seed
                    if not seeded: # Writing first seed
                        text_lst.append(first_act['text'])

                        line['id'] = 'context'
                        line['text'] = '\n'.join(text_lst)
                        line['labels'] = [second_act['text']]
                        line['partner_context'] = '\n'.join(partner_context) if partner_context else ''
                        for k, v in subtask_context.items(): line[k] = v
                        out.append(line)
                        if debug: print('*line*', line, '\n'); input()
                        text_lst = []
                        seeded = True
                else: # utterance
                    if first_act['decision'] == '1':
                        line['id'] = first_act['id']
                        text_lst.append(first_act['text'])
                        line['context1_dataset'] = subtasks[j]
                    if second_act['decision'] == '1':
                        line['labels'] = [second_act['text']]
                        line['context2_dataset'] = subtasks[j]
                    first_verdict = first_act['verdict'].split(',')
                    second_verdict = second_act['verdict'].split(',')
                    suggestions1 = [[t, str(round(v, 2))] for t, v in first_act['beam_texts']]
                    suggestions2 = [[t, str(round(v, 2))] for t, v in second_act['beam_texts']]
                    for k in range(len(first_verdict)):
                        line[f'suggestion1_{subtask}_{k}'] = ' '.join(suggestions1[k]) + f' (verdict: {first_verdict[k]})'
                        line[f'suggestion2_{subtask}_{k}'] = ' '.join(suggestions2[k]) + f' (verdict: {second_verdict[k]})' 
                if debug: print(first_act); print(second_act); input()
            # In case of utterances, we collect the best while iteration.
            if second_act['id'] != 'context' and second_act['id'] != 'seed':
                line['text'] = '\n'.join(text_lst)
                out.append(line)
                if debug: print('*line*', line, '\n'); input()
                text_lst = []
                # TODO add random sampling for response candidates
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def write(self, outfile, world, subtasks, file_format='conversations', indent=4):
        if file_format == 'conversations':
            self.write_conversations_format(outfile, world)
        else:
            # ParlAI text format
            self.write_parlai_format(outfile, subtasks)

    def write_parlai_format(self, outfile, subtasks):
        logging.info(f'Saving log to {outfile} in ParlAI format')
        ana_path = './data/pbst/machine_analysis.txt'
        with PathManager.open(outfile, 'w+') as fw, PathManager.open(ana_path, 'w+') as afw:
            for episode in tqdm(self._logs):
                ep = self.convert_to_labeled_data(episode, subtasks)
                for act in ep:
                    txt = msg_to_str(act)
                    fw.write(txt + '\n')
                fw.write('\n')

                ep = self.convert_to_labeled_data2(episode, subtasks)
                for act in ep:
                    txt = msg_to_str(act)
                    for t in txt.split('\t'):
                        if t.startswith('id:'):
                            afw.write('\n')
                        afw.write('-> '+ t + '\n')
                afw.write('\n')

    def write_conversations_format(self, outfile, world):
        logging.info(f'Saving log to {outfile} in Conversations format')
        Conversations.save_conversations(
            self._logs,
            outfile,
            world.opt,
            self_chat=world.opt.get('selfchat_task', False),
        )