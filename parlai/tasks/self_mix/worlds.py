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
import torch
import numpy as np

from icecream import ic
from parlai.scripts.eval_model import eval_model
from parlai.core.agents import create_agent, create_agent_from_model_file
import os

ROBERTA = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').cuda()

def load_openers(opt) -> Optional[List[str]]:
    base_task = opt['task'].split(':')[0]
    if base_task == 'self_mix':
        return None

    print('[ loading conversation openers... ]')
    # create dummy task so we can get openers from the data
    task_opt = copy.deepcopy(opt)
    task_opt['task'] = base_task + ':SelfmixTeacher'

    # default train will loop forever, but evalmode will stop after one epoch
    datatype = task_opt['datatype']
    print('datatype of pbst task', datatype)
    if 'train' in datatype and 'evalmode' not in datatype:
        task_opt['datatype'] = f'{datatype}:evalmode'
    print(f"[ loading openers in datatype {task_opt['datatype']}")
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
        self.agents, self.retrieval_experts = agents
        super().__init__(opt, self.agents, shared)
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
        # for (leader, follower) in self.agents:
        #     leader.reset()
        #     follower.reset()
        for team in self.agents:
            for agent in team:
                agent.reset()
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
        debug = False

        if self.episode_done():
            self.turn_cnt = 0
            self.episode_cnt += 1
            self.contexts = None
            self.seed_utterances = None
            self.dialogue_history = []
            agents = self.get_agents()
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    agents[i][j].reset()
            for i in range(len(self.retrieval_experts)):
                self.retrieval_experts[i].reset()

        if debug:
            print('\nepisode_cnt', self.episode_cnt)
            print('turn_cnt', self.turn_cnt)
            # print('agents', self.agents)

        if self.turn_cnt == 0:
            self.dialogue_history = []
            self.acts = [[None] * 2 for _ in range(self.nsubtask)]
            # get the beginning of the conversation, which can include contexts
            # and/or any number of starting messages
            self.contexts = np.array(self.get_contexts(self.episode_cnt))
            self.seed_utterances = self._get_seed_utt_acts(
                self.episode_cnt, self.agents
            )
        if self.contexts is not None:
            assert len(self.contexts) == self.nsubtask

            # for i, pair in enumerate(self.contexts):
            #     for j, context in enumerate(pair):
            #         print(f'context[{i}][{j}]', context)

            # initial context
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    context = Message(
                        {'text': self.contexts[i][j], 'episode_done': False, 'id': 'context'} # self.contexts[i][j] = skill context
                    )
                    self.acts[i][j] = context
                    self.agents[i][j].observe(validate(context))
                    if debug: print(f'context[{i}][{j}]', self.acts[i][j]) 
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
                    # if len(utts[0]) > j: # 2 > [0, 1] <- j is always smaller than 2
                    #     self.acts[i][j] = utts[i][j] # [0, 1, 2][0, 1]
                    #     if hasattr(self.agents[i][j], 'self_observe'):
                    #         self.agents[i][j].observe({'episode_done': False})
                    #         self.agents[i][j].self_observe(self.acts[i][j]) # this observes it's own uttereance
                    # else:
                    #     self.acts[i][j] = self.agents[i][j].act() # if j is bigger than 2, but never happens
                    # self.agents[i][1 - j].observe(validate(self.acts[i][j])) # observing the opponent's seed utterance

                    # TODO 버그의 냄새가 난다. self.agents의 observe 로그를 볼수 있나? 확인해서 seed 잘 봤는지 봐야한다.
                    self.acts[i][j] = utts[i][j]
                    if j == 0:
                        if hasattr(self.agents[i][j], 'self_observe'):
                            self.agents[i][j].observe({'episode_done': False})
                            self.agents[i][j].self_observe(self.acts[i][j]) # this observes it's own uttereance
                        else:
                            self.acts[i][j] = self.agents[i][j].act() 
                        self.agents[i][1 - j].observe(validate(self.acts[i][j])) # observing the opponent's seed utterance
                    else:
                        self.agents[i][1 - j].observe(validate(self.acts[i][j])) # observing the opponent's seed utterance
                        if hasattr(self.agents[i][j], 'self_observe'):
                            self.agents[i][j].observe({'episode_done': False})
                            self.agents[i][j].self_observe(self.acts[i][j]) # this observes it's own uttereance
                        else:
                            self.acts[i][j] = self.agents[i][j].act() 
                    
                    if debug: print(f'seed uttererance pairs [{i}][{j}]', self.acts[i][j])

            self.dialogue_history.append(self.acts[0][0])
            self.dialogue_history.append(self.acts[0][1])

            if debug:
                print('\nContexts and seeds are initialized. Starting bot2bot conversation.\n')
            # TODO ED의 context가 이상하다. -> opener나 alignment가 이상하다
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents

            # if debug: 
            #     for i in range(len(self.agents)):
            #         for j in [0, 1]:
            #             print(f'self.acts[{i}][{j}]', self.acts[i][j]['text'])    
            #     input('acts initialized')

            # Leaders action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][0] = agents[i][0].act()
                # the number of predicted beam may not reach to the given beam size
                if len(acts[i][0]['beam_texts']) < self.opt['beam_size']:
                    beam_texts = copy.deepcopy(acts[i][0]['beam_texts'])
                    for j in range(len(acts[i][0]['beam_texts']), self.opt['beam_size'] ):
                        beam_texts.append(('', -1e4))
                    acts[i][0].force_set('beam_texts', beam_texts)
                response_candidates.append(acts[i][0]['beam_texts'])
            if debug: input('Leaders actioned\n')
            
            # Leaders debate
            verdicts = self.filter_out(response_candidates, self.documents[:,0]) # TODO provide only leader's context
            decisions, (expertise_id, beam_id) = self.decide(response_candidates, verdicts, self.documents[:,0])
            for i, (verdict, decision) in enumerate(zip(verdicts, decisions)):
                acts[i][0].force_set('verdict', ','.join(list(map(str, verdict))))
                acts[i][0].force_set('decision', ','.join(list(map(str, decision))))
            if debug: input('Leaders debated\n')

            if debug:
                for i in range(len(self.agents)): print(f'self.acts[{i}][0]', self.acts[i][0], end='\n\n')
                input('Before decision updates\n')

            acts[expertise_id][0].force_set('text', response_candidates[expertise_id][beam_id][0])

            if debug: 
                print(f'\n ***The final decision is :{acts[expertise_id][0]["text"]}***\n')
                print('Decision Matrix', decisions)
                input('Updated leaders\' decision')
                for i in range(len(self.agents)): print(f'self.acts[{i}][0]', self.acts[i][0], end='\n\n')
                input('After decision updates\n')
                
            # Followers observe
            for i in range(len(self.agents)):
                agents[i][1].observe(validate(acts[expertise_id][0]))
                self.dialogue_history.append(acts[expertise_id][0])
            if debug: input('Followers observed\n')
            
            ### ================== turn switching =================== ###
            
            # Followers action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][1] = agents[i][1].act()
                # the number of predicted beam may not reach to the given beam size
                if len(acts[i][1]['beam_texts']) < self.opt['beam_size']:
                    beam_texts = copy.deepcopy(acts[i][1]['beam_texts'])
                    for j in range(len(acts[i][1]['beam_texts']), self.opt['beam_size'] ):
                        beam_texts.append(('', -1e4))
                    acts[i][1].force_set('beam_texts', beam_texts)
                response_candidates.append(acts[i][1]['beam_texts'])
            if debug: input('Followers actioned\n')

            # Followers debate
            verdicts = self.filter_out(response_candidates, self.documents[:,1])
            decisions, (expertise_id, beam_id) = self.decide(response_candidates, verdicts, self.documents[:,1])
            for i, (verdict, decision) in enumerate(zip(verdicts, decisions)):
                acts[i][1].force_set('verdict', ','.join(list(map(str, verdict))))
                acts[i][1].force_set('decision', ','.join(list(map(str, decision))))
            if debug: input('Followers debated\n')
            
            if debug: 
                for i in range(len(self.agents)): print(f'self.acts[{i}][1]', self.acts[i][1], end='\n\n')
                input('Before decision updates\n')
            
            acts[expertise_id][1].force_set('text', response_candidates[expertise_id][beam_id][0])

            if debug:
                print(f'\n ***The final decision is :{acts[expertise_id][1]["text"]}***\n')
                print('Decision Matrix', decisions)
                input('Updated followers\' decision')
                for i in range(len(self.agents)): print(f'self.acts[{i}][1]', self.acts[i][1], end='\n\n')
                input('After decision updates\n')
                
            # Leaders observe
            for i in range(len(self.agents)):
                agents[i][0].observe(validate(acts[expertise_id][1]))
                self.dialogue_history.append(acts[expertise_id][1])
            if debug: input('Leaders observed\n')

        self.update_counters()
        self.turn_cnt += 1

    def fact_check(self, premise, hypotheis) -> bool:
        # sentence-pair NLI for fact check
        ROBERTA.eval()
        with torch.no_grad():
            # label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
            tokens = ROBERTA.encode(premise, hypotheis)
            score = ROBERTA.predict('mnli', tokens)
            prediction = score.argmax().item()
        
        if prediction == 0:
            verdict = 0
        else:
            verdict = 1

        return verdict

    def filter_out(self, response_candidates, contexts):
        debug = False
        short = False

        ntask = len(response_candidates)
        nbeam = len(response_candidates[0])
        virdicts = [[1] * nbeam for _ in range(ntask)]

        
        subtasks = self.opt.get('subtasks')
        
        # Cross-domain first, regarding them as more reliable filtering than cross-claim fact-checking
        if debug or short: cnt = 0
        # for i, context_pair in enumerate(contexts):
        #     for j, context in enumerate(context_pair): # TODO Decide whether we comapare a claim to both leading/following contexts
        for i, context in enumerate(contexts):
            for m, beam_texts in enumerate(response_candidates):
                for n, (claim, _score) in enumerate(beam_texts):
                    if debug or short: cnt += 1
                    if virdicts[m][n]:
                        if debug: print('FACT CHECKING CNT', cnt, f'{subtasks[i]} {subtasks[m]}-{n}')
                        if debug: print('Context', context)
                        if debug: print('Claim', claim)
                        # if context == '' or claim == '': # TODO context가 없다고 claim이 contradiction이라는 것은 이상하다. 확인 필요함.
                        if claim == '': # TODO 그렇다고 empty premise에 대한 claim의 hypothesis를 보는 것도 웃기다. 확인 필요함.
                            virdicts[m][n] = 0
                            continue
                        if i == m:
                            if debug: print()
                            continue
                        if debug: print('Virdict', virdicts[m][n])
                        if debug: print()
                        virdicts[m][n] &= self.fact_check(context, claim)
                        if short: 
                            # if virdicts[m][n] == False: print(f'Cross-domain contradiction on fact-checking #{cnt}\nContext - {subtasks[i]} #{j+1}:\n{context}\nClaim - {subtasks[m]} #{n+1}:\n\t{claim} => Filtered Out\n')
                            if virdicts[m][n] == 0: print(f'Cross-domain contradiction on fact-checking #{cnt}\nContext - {subtasks[i]}:\n{context}\nClaim - {subtasks[m]} #{n+1}:\n\t{claim} => Filtered Out\n')
        if debug or short: print('Virdicts after cross-domain', virdicts, '\n')
        
        # Cross-claims
        # if debug or short: cnt = 0
        # for i, beam_texts1 in enumerate(response_candidates):
        #     for j, (claim1, _score1) in enumerate(beam_texts1):
        #         for m, beam_texts2 in enumerate(response_candidates):
        #             for n, (claim2, _score2) in enumerate(beam_texts2):
        #                 if debug or short: cnt += 1
        #                 if virdicts[m][n]:
        #                     if debug: print('FACT CHECKING CNT', cnt, f'{subtasks[i]}-{j} {subtasks[m]}-{n}')
        #                     if debug: print('Claim1', claim1)
        #                     if debug: print('Claim2', claim2)
        #                     if claim1 == '' or claim2 == '':
        #                         virdicts[m][n] = 0
        #                         continue
        #                     if i == m:
        #                         if debug: print()
        #                         continue
        #                     if debug: print('Virdict', virdicts[m][n])
        #                     if debug: print()
        #                     virdicts[m][n] &= self.fact_check(claim1, claim2)
        #                     if short: 
        #                         if virdicts[m][n] == 0: print(f'Cross-claim contradiction on fact-checking #{cnt}\nClaim1 - {subtasks[i]} #{j+1}:\n{claim1}\nClaim2 - {subtasks[m]} #{n+1}:\n\t{claim2} => Filtered Out\n')
        # if debug or short: print('Virdicts after cross-claim', virdicts, '\n')

        # Getting the agreements for dialogue consistency
        if debug or short:
            agreements = [response_candidates[i][j][0] for i, task in enumerate(virdicts) for j, virdic in enumerate(task) if virdic == 1]
            print("\nExpert's Agreements")
            print('\n'.join(agreements)) if agreements else print('None')
            print()

        return virdicts

    def decide(self, response_candidates, virdicts, documents):
        num_agents = len(response_candidates)
        beam_size = len(response_candidates[0])

        # Create response candidate file for fixed candidate
        response_candidates_list = []
        f = open(self.opt.get('datapath') + '/response_candidates.txt', 'w')
        for i in range(num_agents):
            for j in range(beam_size):
                f.write(response_candidates[i][j][0] + '\n')
                response_candidates_list.append(response_candidates[i][j][0])
        f.close()

        # skill-aware rankers observe history and sCtx
        retrieval_results = []
        for i in range(num_agents):
            context = Message({'text': documents[i], 'episode_done': False, 'id': 'context'})

            self.retrieval_experts[i].set_fixed_candidates(False)
            self.retrieval_experts[i].observe(validate(context))
            for msg in self.dialogue_history:
                self.retrieval_experts[i].observe(validate(msg))
            retrieval_results.append(self.retrieval_experts[i].act()['text_candidates'])

        score = virdicts
        max_score = -1

        for i in range(num_agents):
            for j in range(beam_size):
                score[i][j] = score[i][j] * ((retrieval_results[0].index(response_candidates[i][j][0]) + 1) + (retrieval_results[1].index(response_candidates[i][j][0]) + 1) + (retrieval_results[2].index(response_candidates[i][j][0]) + 1))
                if score[i][j] > max_score:
                    max_score = score[i][j]
                    max_row = i
                    max_col = j

        decimat = np.zeros_like(virdicts)
        decimat[max_row][max_col] = 1

        return decimat, (max_row, max_col)