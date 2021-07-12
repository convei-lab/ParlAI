#! /usr/bin/bash
set -o xtrace
# # Conv
# CUDA_VISIBLE_DEVICES=1 parlai eval_model \
# -mf zoo:dodecadialogue/convai2_ft/model -t blended_skill_talk \
# --skip-generation False --inference nucleus --topp 0.9 --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3

# # WOW
# CUDA_VISIBLE_DEVICES=1 parlai eval_model \
# -mf zoo:dodecadialogue/wizard_of_wikipedia_ft/model -t blended_skill_talk \
# --skip-generation False --inference nucleus --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3

# # ED
# CUDA_VISIBLE_DEVICES=1 parlai eval_model \
# -mf zoo:dodecadialogue/empathetic_dialogues_ft/model -t blended_skill_talk \
# --skip-generation False --inference nucleus --topp 0.9 --beam-size 3 --beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3

