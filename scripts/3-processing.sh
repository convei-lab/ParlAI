#!/bin/bash
set -o xtrace
cd ..

# We could look at that data using the usual display data script
# python parlai/scripts/display_data.py --task fromfile:parlaiformat --fromfile_datapath /home/theorist/data3/ParlAI/data/pbst/archived/pbst-parlai-format.txt

# In conversations format (JSONL format)
# python parlai/scripts/display_data.py --task jsonfile --jsonfile-datapath /home/theorist/data3/ParlAI/data/pbst/archived/pbst-conversations-format.jsonl


# python parlai/scripts/display_data.py --task pbst

# parlai self_chat --model-file zoo:dodecadialogue/empathetic_dialogues_ft/model --task pbst \
parlai self_chat --model-file zoo:dodecadialogue/empathetic_dialogues_ft/model --task blended_skill_talk \
                --num-self-chats 10 --display-examples True --datatype valid \
                --skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
                --partner-model-file zoo:dodecadialogue/wizard_of_wikipedia_ft/model \
                --partner-opt-file data/models/dodecadialogue/wizard_of_wikipedia_ft/model.opt \
                --verbose --outfile ./data/pbst/pbst.json --save-format conversations --include-initial-utterances 1
