#!/bin/bash
set -o xtrace
cd ..

parlai self_mix --expert-model-files \
zoo:dodecadialogue/convai2_ft/model,\
zoo:dodecadialogue/wizard_of_wikipedia_ft/model,\
zoo:dodecadialogue/empathetic_dialogues_ft/model \
--expert-model-opt-files \
/home/theorist/data3/ParlAI/data/models/dodecadialogue/convai2_ft/model.opt,\
/home/theorist/data3/ParlAI/data/models/dodecadialogue/wizard_of_wikipedia_ft/model.opt,\
/home/theorist/data3/ParlAI/data/models/dodecadialogue/empathetic_dialogues_ft/model.opt \
--num-self-mixs 10 \
--display-examples True --datatype valid \
--outfile ./data/pbst/machine_generated.txt --verbose \
--subtasks convai2,wizard_of_wikipedia,empatheticdialogues \
--task pbst --seed_messages_from_task 1 \
--model-file zoo:dodecadialogue/convai2_ft/model \
--skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
--save-format parlai
# --save-format conversations
# --partner-model-file zoo:dodecadialogue/wizard_of_wikipedia_ft/model \
# --partner-opt-file data/models/dodecadialogue/wizard_of_wikipedia_ft/model.opt \