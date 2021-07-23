#!/bin/bash

set -o xtrace
cd ..
parlai display_data --task persona_inference
parlai display_data --task topic_inference
parlai display_data --task emotion_inference

parlai train_model -m tfidf_retriever -t persona_inference -mf ./data/pbst/convai2/tfidf_retriever/model -dt train:ordered -eps 1
parlai train_model -m tfidf_retriever -t topic_inference -mf ./data/pbst/wizard_of_wikipedia/tfidf_retriever/model -dt train:ordered -eps 1
parlai train_model -m tfidf_retriever -t emotion_inference -mf ./data/pbst/empatheticdialogues/tfidf_retriever/model -dt train:ordered -eps 1