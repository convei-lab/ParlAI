#!/bin/bash
set -o xtrace
parlai train_model -t blended_skill_talk -m image_seq2seq -mf /tmp/model

