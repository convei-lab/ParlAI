import parlai.core.build_data as build_data
import os
import json
from tqdm import tqdm
from icecream import ic

def build(opt):
    dpath = os.path.join(opt['datapath'], 'emotion_inference')
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

        seed_utterance_empathy_path = '/home/minju/kbs/seed_utterance_pairs_empatheticdialogues.json'
        f_empathy_path = '/home/minju/kbs/following_empatheticdialogues_kb.json'
        l_empathy_path = '/home/minju/kbs/leading_empatheticdialogues_kb.json'

        # file structure: utterance, label, candidates(splitted by '\t')
        emotion_inference_train_list = []
        emotion_inference_valid_list = []
        emotion_inference_test_list = []
        with open(seed_utterance_empathy_path) as json_file:
            seed_utterance_empathy = json.load(json_file)

        # only topic (cause we can get passage by retrieving)
        with open(l_empathy_path) as json_file:
            l_empathy = json.load(json_file)

        l_empathy_unique = list(set(l_empathy))

        for i in tqdm(range(len(seed_utterance_empathy))):
            dialog_dict_l = {}
            dialog_dict_l['text'] = seed_utterance_empathy[i][0]
            dialog_dict_l['labels'] = l_empathy[i]
            dialog_dict_l['label_candidates'] = l_empathy_unique

            if i <= int(len(seed_utterance_empathy) * 0.3):
                emotion_inference_train_list.append(dialog_dict_l)
            elif i > int(len(seed_utterance_empathy) * 0.8) and i < int(len(seed_utterance_empathy) * 0.9):
                emotion_inference_valid_list.append(dialog_dict_l)
            elif i > int(len(seed_utterance_empathy) * 0.9):
                emotion_inference_test_list.append(dialog_dict_l)
  
        f = open(dpath + '/fixed_candidates.txt', 'w')
        for candidate in l_empathy_unique:
            f.write(candidate + '\n')
        print("Saved candidate file at", dpath + '/fixed_candidates.txt')       
        f.close()      

        
        with open(dpath + '/train.json', "w") as json_file:
            json.dump(emotion_inference_train_list, json_file)
        print("Saved file at", dpath + '/train.json')

        # Due to storge issue, use train file as valid/test file
        with open(dpath + '/valid.json', "w") as json_file:
            json.dump(emotion_inference_valid_list, json_file)
        print("Saved file at", dpath + '/valid.json')

        with open(dpath + '/test.json', "w") as json_file:
            json.dump(emotion_inference_test_list, json_file)
        print("Saved file at", dpath + '/test.json')        

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)