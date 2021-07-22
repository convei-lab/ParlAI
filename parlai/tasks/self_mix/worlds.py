#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from typing import Any, Dict, List, Optional

from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, TeamDebateWorld, validate
from parlai.core.message import Message


def load_openers(opt) -> Optional[List[str]]:
    base_task = opt['task'].split(':')[0]
    if base_task == 'self_mix':
        # TODO(#2284): Load default openers from s3
        return None

    print('[ loading conversation openers... ]')
    # create dummy task so we can get openers from the data
    task_opt = copy.deepcopy(opt)
    task_opt['task'] = base_task

    # default train will loop forever, but evalmode will stop after one epoch
    datatype = task_opt['datatype']
    if 'train' in datatype and 'evalmode' not in datatype:
        task_opt['datatype'] = f'{datatype}:evalmode'
    task_opt['interactive_task'] = False
    task_opt['selfchat_task'] = False
    task_opt['selfmix_task'] = False
    task_opt['fixed_response'] = None
    task_opt['seed_messages_from_task'] = False
    task_agent = FixedResponseAgent(task_opt)
    task_world = create_task(task_opt, task_agent)

    # run through task data, collecting all first messages
    # openers = set()
    openers = []
    is_first_turn = True
    while not task_world.epoch_done():
        task_world.parley()
        msg = task_world.get_acts()[0]
        # add only the first message in the episode
        if is_first_turn and msg.get('text') and msg.get('eval_labels'):
            openers.append([msg['text'], msg['eval_labels'][0]])
        is_first_turn = msg.get('episode_done', False)
    # for opener in openers:
    #     print('opener', opener)
    print(f'[ loaded {len(openers)} openers ]')
    return openers


def load_openers_from_file(filepath: str) -> List[str]:
    openers = []
    with open(filepath, 'r') as f:
        openers = [l.strip() for l in f]
    return openers


class SelfMixWorld(TeamDebateWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts(shared=shared)
        self._openers = None
        self.init_openers()
        self.max_turn_cnt = self.opt.get('selfmix_max_turns', 10)
        self.turn_cnt = 0
        self.episode_cnt = 0
        self.nsubtask = len(self.opt.get('subtasks'))
        self.world_name = 'SelfMixWorld'

    def init_contexts(self, shared=None) -> None:
        """
        Override to load or instantiate contexts to be used to seed the self mix.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the self mix episode.

        This function will be called before the first turn of every episode.
        """
        return ['Hi!', '']

    def init_openers(self) -> None:
        """
        Override to load or instantiate opening messages to be used to seed the self
        mix.
        """
        if self.opt.get('seed_messages_from_task'):
            self._openers = load_openers(self.opt)
        elif self.opt.get('seed_messages_from_file'):
            self._openers = load_openers_from_file(self.opt['seed_messages_from_file'])

    def get_openers(self, episode_num: int) -> Optional[List[str]]:
        """
        Override to return one or more opening messages with which to seed the self mix
        episode.

        The return value should be an array of strings, each string being a message in
        response to the string before it.
        """

        if self._openers:
            return [open_msg for open_msg in self._openers[episode_num]]# [random.choice(self._openers)]
        return None

    def reset(self):
        """
        Reset all agents in the world, and world statistics.
        """
        for (leader, follower) in self.agents:
            leader.reset()
            follower.reset()
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time.reset()

    # def write(self):
    #     pass

    def display(self):
        s = super().display()
        if self.turn_cnt == 0:
            s += '\n==============================\n'
        return s

    def episode_done(self):
        return self.turn_cnt >= self.max_turn_cnt

    def _get_seed_utt_acts(
        self, episode_num: int, agents: List[Agent]
    ) -> List[Dict[str, Any]]:
        # def make_agent_action(utterance: str, agentss: Agent) -> Dict[str, Any]:
        #     agent_action = []
        #     for i, a_pair in enumerate(agentss):
        #         a_action_pair = []
        #         for a in a_pair:
        #             a_action_pair.append({'text': utterance[i], 'episode_done': False, 'id': a.id})
        #         agent_action.append(a_action_pair)
        #     return agent_action

        openers = self.get_openers(episode_num)
        if not openers:
            return []
        
        # agent_action = []
        # for agent in agents:
        #     paired_action = []
        #     for i in [0, 1]:
        #         paired_action.append(make_agent_action(openers[0], agent[i]))
        #     agent_action.append(paired_action)
        # return agent_action
        # actions = list(map(make_agent_action, openers, agents))

        actions = []
        for (leader, follower) in agents:
            action = [{'text': openers[0], 'episode_done': False, 'id': 'seed'},
                      {'text': openers[1], 'episode_done': False, 'id': 'seed'}]
            actions.append(action)
        return actions   

    def parley(self):
        debug = False # TODO decide this
        subtasks = ['convai2', 'wizardofwikipedia', 'emphatheticdialogues']

        if self.episode_done():
            self.turn_cnt = 0
            self.episode_cnt += 1
            self.contexts = None
            self.seed_utterances = None
            agents = self.get_agents()
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    agents[i][j].reset()

        if debug:
            print('episode_cnt', self.episode_cnt)
            print('turn_cnt', self.turn_cnt)
            # print('agents', self.agents)

        if self.turn_cnt == 0:
            self.acts = [[None] * 2 for _ in range(self.nsubtask)]
            # get the beginning of the conversation, which can include contexts
            # and/or any number of starting messages
            self.contexts = self.get_contexts(self.episode_cnt)
            self.seed_utterances = self._get_seed_utt_acts(
                self.episode_cnt, self.agents
            )
        if self.contexts:
            assert len(self.contexts) == self.nsubtask

            # for i, pair in enumerate(self.contexts):
            #     for j, context in enumerate(pair):
            #         print(f'context[{i}][{j}]', context)

            # initial context
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    context = Message(
                        {'text': self.contexts[i][j], 'episode_done': False, 'id': 'context'}
                    )
                    self.acts[i][j] = context
                    self.agents[i][j].observe(validate(context))
                    if debug: print(f'self.act[{i}][{j}]', self.acts[i][j]) 
            # clear contexts so they are only added once per episode
            self.documents = copy.deepcopy(self.contexts)
            self.contexts = None

        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            assert len(self.seed_utterances) == self.nsubtask
            utts = self.seed_utterances
            # print('seed utt', utts)
            self.seed_utterances = None
            # print('self.seed_utt', self.seed_utterances)

            # for i, seed_utt in enumerate(self.seed_utterances):
            #     print(f'seed_utterance{i}', seed_utt)

            # process the turn
            for i in range(len(self.agents)):
                for j in [0, 1]: # leading utterance and following utterance
                # as we have a seed utterance, add it to the conversation
                    if len(utts[0]) > j:
                        self.acts[i][j] = utts[i][j]
                        if hasattr(self.agents[i][j], 'self_observe'):
                            self.agents[i][j].observe({'episode_done': False})
                            self.agents[i][j].self_observe(self.acts[i][j])
                    else:
                        self.acts[i][j] = self.agents[i][j].act()
                    self.agents[i][1 - j].observe(validate(self.acts[i][j]))
                    if debug: print(f'self.acts[{i}][{j}]', self.acts[i][j])
            if debug:
                input('Episode initialized\n')
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents

            # if debug: 
            # for i in range(len(self.agents)):
            #     for j in [0, 1]:
            #         print(f'self.acts[{i}][{j}]', self.acts[i][j]['text'])    
            # input('acts initialized')

            # Leaders action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][0] = agents[i][0].act()
                response_candidates.append(acts[i][0]['beam_texts'])

            # Leaders debate
            verdict = filter_out(response_candidates, self.documents)
            decision = decide_action(response_candidates, verdict)
            for i in range(len(self.agents)):
                acts[i][0].force_set('decision', '0')
                acts[i][0].force_set('verdict', ','.join([str(int(v)) for v in verdict[i]]))
            acts[decision][0].force_set('decision', '1')

            # Followers observe
            for i in range(len(self.agents)):
                agents[i][1].observe(validate(acts[decision][0]))
            
            if debug: 
                for i in range(len(self.agents)):
                    for j in [0]:
                        print(f'self.acts[{i}][{j}]', self.acts[i][j])
                print(f'Decision: {decision}', acts[decision][0])
                input('Leaders actioned & followers observed\n')
                
            # Followers action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][1] = agents[i][1].act()
                response_candidates.append(acts[i][1]['beam_texts'])

            # Followers debate
            verdict = filter_out(response_candidates, self.documents)
            decision = decide_action(response_candidates, verdict)
            for i in range(len(self.agents)):
                acts[i][1].force_set('decision', '0')
                acts[i][1].force_set('verdict', ','.join([str(int(v)) for v in verdict[i]]))
            acts[decision][1].force_set('decision', '1')

            # leaders observe
            for i in range(len(self.agents)):
                # agents[i][0].observe(validate(decision))
                agents[i][0].observe(validate(acts[decision][1]))
            
            if debug: 
                for i in range(len(self.agents)):
                    for j in [1]:
                        print(f'self.acts[{i}][{j}]', self.acts[i][j])
                print(f'Decision: {decision}', acts[decision][1])
                input('Followers actioned & leaders observed\n')
        self.update_counters()
        self.turn_cnt += 1

def fact_check(claim, doc) -> bool:
    # TODO sentence-pair NLI for fact check
    # random decision
    verdict = True if random.randint(1, 10) >= 7 else False
    return verdict

def filter_out(response_candidates, contexts):
    debug = False

    ntask = len(response_candidates)
    nbeam = len(response_candidates[0])
    virdicts = [[True] * nbeam for _ in range(ntask)]

    # cross-claims
    for i, beam_texts in enumerate(response_candidates):
        for j, claim in enumerate(beam_texts):
            for m, beam_texts in enumerate(response_candidates):
                for n, doc in enumerate(beam_texts):
                    if i == m and j == n:
                        continue
                    virdicts[i][j] = fact_check(claim, doc)
    if debug: print('After cross-claim', virdicts)

    # cross-domain
    for i, beam_texts in enumerate(response_candidates):
        for j, claim in enumerate(beam_texts):
            for m, context_pair in enumerate(contexts):
                for n, context in enumerate(context_pair):
                    if i == m or not context:
                        continue
                    virdicts[i][j] &= fact_check(claim, context)
    if debug: print('After cross-domain', virdicts)
    
    # guard no suggestion from a certain expertise
    for i, beam_texts in enumerate(response_candidates):
        for j, claim in enumerate(beam_texts):
            if not any([virdicts[i][j] for i in range(ntask) for j in range(nbeam)]):
                virdicts[i][0] = True
                break

    if debug: print('Guarding cross-outs', virdicts)

    return virdicts

def decide_action(actions, virdicts):

    # TODO somehow select 
    decision = random.randint(0, len(virdicts)-1)

    # for i in range(len(virdicts[decision])):
    #     if virdicts[i]:
    #         final_response = actions[i] #  only the highest-scoring text (never a beam_texts, for now)
    #         break


    return decision