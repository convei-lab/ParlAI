#!/bin/bash
set -o xtrace
parlai train_model -t pbst -m image_seq2seq -mf /tmp/model --num_epochs 1

