#!/bin/bash
set -o xtrace
cd ..

parlai self_mix \
--subtasks convai2,wizard_of_wikipedia,empatheticdialogues \
--num-self-mixs 100 \
--selfmix-max-turns 6 \
--datatype train \
--expert-model-files zoo:dodecadialogue/convai2_ft/model,zoo:dodecadialogue/wizard_of_wikipedia_ft/model,zoo:dodecadialogue/empathetic_dialogues_ft/model \
--expert-model-opt-files scripts/conv.opt,scripts/wow.opt,scripts/ed.opt \
--display-examples True \
--task pbst --seed_messages_from_task 1 \
--model-file zoo:dodecadialogue/convai2_ft/model \
--skip-generation False --inference nucleus \
--beam-size 3 \
--beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
--save-format parlai \
--outfile /home/minju/bst/pbst_files/file.txt \
--ranker-model-files zoo:pretrained_transformers/model_poly/model,/home/minju/empathetic_dialogues_poly/model.checkpoint,/home/minju/wizard_of_wikipedia_poly/model.checkpoint

# parlai self_mix \
# --subtasks convai2,wizard_of_wikipedia,empatheticdialogues \
# --num-self-mixs 500 \
# --selfmix-max-turns 6 \
# --datatype valid \
# --expert-model-files zoo:dodecadialogue/convai2_ft/model,zoo:dodecadialogue/wizard_of_wikipedia_ft/model,zoo:dodecadialogue/empathetic_dialogues_ft/model \
# --expert-model-opt-files scripts/conv.opt,scripts/wow.opt,scripts/ed.opt \
# --display-examples True \
# --task pbst --seed_messages_from_task 1 \
# --model-file zoo:dodecadialogue/convai2_ft/model \
# --skip-generation False --inference nucleus \
# --beam-size 3 \
# --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
# --save-format parlai \
# --ranker-model-files zoo:pretrained_transformers/model_poly/model,/home/minju/empathetic_dialogues_poly/model.checkpoint,/home/minju/wizard_of_wikipedia_poly/model.checkpoint

# parlai self_mix \
# --subtasks convai2,wizard_of_wikipedia,empatheticdialogues \
# --num-self-mixs 500 \
# --selfmix-max-turns 6 \
# --datatype test \
# --expert-model-files zoo:dodecadialogue/convai2_ft/model,zoo:dodecadialogue/wizard_of_wikipedia_ft/model,zoo:dodecadialogue/empathetic_dialogues_ft/model \
# --expert-model-opt-files scripts/conv.opt,scripts/wow.opt,scripts/ed.opt \
# --display-examples True \
# --task pbst --seed_messages_from_task 1 \
# --model-file zoo:dodecadialogue/convai2_ft/model \
# --skip-generation False --inference nucleus \
# --beam-size 3 \
# --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
# --save-format parlai \
# --ranker-model-files zoo:pretrained_transformers/model_poly/model,/home/minju/empathetic_dialogues_poly/model.checkpoint,/home/minju/wizard_of_wikipedia_poly/model.checkpoint