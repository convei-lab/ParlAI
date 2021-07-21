import parlai.core.build_data as build_data
import os
import json
from tqdm import tqdm
from icecream import ic

def build(opt):
    dpath = os.path.join(opt['datapath'], 'topic_inference')
    version = 'v0.0'

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        seed_utterance_convai2_path = '/home/minju/kbs/seed_utterance_pairs_convai2.json'
        f_convai2_path = '/home/minju/kbs/following_convai2_kb.json'
        l_convai2_path = '/home/minju/kbs/leading_convai2_kb.json'

        seed_utterance_wow_path = '/home/minju/kbs/seed_utterance_pairs_wizard_of_wikipedia.json'
        f_wow_path = '/home/minju/kbs/following_wizard_of_wikipedia_kb.json'
        l_wow_path = '/home/minju/kbs/leading_wizard_of_wikipedia_kb.json'

        seed_utterance_empathy_path = '/home/minju/kbs/seed_utterance_pairs_wizard_of_wikipedia.json'
        f_empathy_path = '/home/minju/kbs/following_empatheticdialogues_kb.json'
        l_empathy_path = '/home/minju/kbs/leading_empatheticdialogues_kb.json'

        # file structure: utterance, label, candidates(splitted by '\t')
        topic_inference_train_list = []
        topic_inference_valid_list = []
        topic_inference_test_list = []
        with open(seed_utterance_wow_path) as json_file:
            seed_utterance_wow = json.load(json_file)

        # only topic (cause we can get passage by retrieving)
        with open(l_wow_path) as json_file:
            l_wow = json.load(json_file)

        l_wow_unique = list(set(l_wow))

        for i in tqdm(range(len(seed_utterance_wow))):
            dialog_dict_f = {}
            dialog_dict_f['text'] = seed_utterance_wow[i][1]
            dialog_dict_f['labels'] = l_wow[i]
            dialog_dict_f['label_candidates'] = l_wow_unique

            dialog_dict_l = {}
            dialog_dict_l['text'] = seed_utterance_wow[i][0]
            dialog_dict_l['labels'] = l_wow[i]
            dialog_dict_l['label_candidates'] = l_wow_unique

            if i <= int(len(seed_utterance_wow) * 0.3):
                topic_inference_train_list.append(dialog_dict_f)
                topic_inference_train_list.append(dialog_dict_l)
            elif i > int(len(seed_utterance_wow) * 0.8) and i < int(len(seed_utterance_wow) * 0.9):
                topic_inference_valid_list.append(dialog_dict_f)
                topic_inference_valid_list.append(dialog_dict_l)
            elif i > int(len(seed_utterance_wow) * 0.9):
                topic_inference_test_list.append(dialog_dict_f)
                topic_inference_test_list.append(dialog_dict_l)
  
        f = open(dpath + '/fixed_candidates.txt', 'w')
        for candidate in l_wow_unique:
            f.write(candidate + '\n')
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

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)