#!/bin/bash
set -o xtrace
cd ..

# tutorial
# parlai self_chat --model-file zoo:pretrained_transformers/model_poly/model --task convai2 \
#                 --inference topk --num-self-chats 10 --display-examples True --datatype valid

# conv
# Selfchat 기본인자
parlai self_chat --model-file zoo:dodecadialogue/convai2_ft/model --task blended_skill_talk \
                --num-self-chats 10 --display-examples True --datatype valid \
                --skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
                --partner-model-file zoo:dodecadialogue/wizard_of_wikipedia_ft/model \
                --partner-opt-file data/models/dodecadialogue/wizard_of_wikipedia_ft/model.opt \
                --verbose --outfile ./data/pbst/pbst.json --save-format conversations 
                # --verbose --outfile ./data/pbst/pbst-conversations-format.json --save-format conversations # JSON 포맷 검증 -> 이게 더 편해서 선택
                # --verbose --outfile ./data/pbst/pbst-parlai-format.txt --save-format parlai # ParlAI 포맷 검증
# # wow
# parlai self_chat --model-file zoo:dodecadialogue/wizard_of_wikipedia_ft/model --task blended_skill_talk \
#                 --num-self-chats 10 --display-examples True --datatype valid \
#                 --skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3

# # ed
# parlai self_chat --model-file zoo:dodecadialogue/empathetic_dialogues_ft/model --task blended_skill_talk \
#                 --num-self-chats 10 --display-examples True --datatype valid \
#                 --skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3

