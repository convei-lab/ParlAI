#!/bin/bash

set -o xtrace
cd ..
parlai display_data --task persona_inference
parlai display_data --task topic_inference
parlai display_data --task emotion_inference

parlai train_model -m tfidf_retriever -t persona_inference -mf ./data/pbst/contextual_alignment/persona_inference/lexical_retrieval/model -dt train:ordered -eps 1 \
--num_epochs 1
parlai train_model -m tfidf_retriever -t topic_inference -mf ./data/pbst/contextual_alignment/topic_inference/lexical_retrieval/model -dt train:ordered -eps 1 \
--num_epochs 1
parlai train_model -m tfidf_retriever -t emotion_inference -mf ./data/pbst/contextual_alignment/emotion_inference/lexical_retrieval/model -dt train:ordered -eps 1 \
--num_epochs 1