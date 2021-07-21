import parlai.core.build_data as build_data
import os
import json
import random
from tqdm import tqdm
from icecream import ic

def build(opt):
    dpath = os.path.join(opt['datapath'], 'persona_inference')
    version = 'v0.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        persona_utterance_dict_path = '/home/minju/bst/persona_link_file/p2u.json'

        # Make train file
        persona_inference_list = []
        with open(persona_utterance_dict_path) as json_file:
            persona_utterance_dict = json.load(json_file)
        persona_list = list(persona_utterance_dict.keys())

        for i in tqdm(range(len(persona_list))):
            utterance_list = persona_utterance_dict[persona_list[i]]
            for utterance in utterance_list:
                train_dict = {}
                train_dict['text'] = utterance
                train_dict['labels'] = persona_list[i]
                train_dict['label_candidates'] = persona_list
                persona_inference_list.append(train_dict)

        random.shuffle(persona_inference_list)
        persona_inference_train_list = persona_inference_list[0:int(len(persona_inference_list) * 0.3)]
        persona_inference_valid_list = persona_inference_list[int(len(persona_inference_list) * 0.8): int(len(persona_inference_list) * 0.9)]
        persona_inference_test_list = persona_inference_list[int(len(persona_inference_list) * 0.9):]

        with open(dpath + '/train.json', "w") as json_file:
            json.dump(persona_inference_train_list, json_file)
        print("Saved file at", dpath + '/train.json')

        # Due to storge issue, use train file as valid/test file
        with open(dpath + '/valid.json', "w") as json_file:
            json.dump(persona_inference_valid_list, json_file)
        print("Saved file at", dpath + '/valid.json')

        with open(dpath + '/test.json', "w") as json_file:
            json.dump(persona_inference_test_list, json_file)
        print("Saved file at", dpath + '/test.json')       

        f = open(dpath + '/fixed_candidates.txt', 'w')
        for persona in persona_list:
            f.write(persona + '\n') 
        print("Saved candidate file at", dpath + '/fixed_candidates.txt')   

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)