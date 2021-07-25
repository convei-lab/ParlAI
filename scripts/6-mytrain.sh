#!/bin/bash
set -o xtrace
# parlai train_model -t pbst -m image_seq2seq -mf /tmp/model --num_epochs 1
# parlai train_model -t pbst -m transformers/polyencoder -mf zoo:pretrained_transformers/model_poly/model \--encode-candidate-vecs true --num_epochs 1
cd ..
parlai train_model \
  --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
  -t pbst \
  --model transformer/polyencoder --batchsize 1 --eval-batchsize 1 \
  --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
  -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
  --text-truncate 360 --num-epochs 1.0 --max_train_time 200000 -veps 0.5 \
  -vme 8000 --validation-metric accuracy --validation-metric-mode max \
  --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 False \
  --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
  --variant xlm --reduction-type mean --share-encoders False \
  --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
  --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
  --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
  --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
  --poly-attention-type basic --dict-endtoken __start__ \
  --model-file ./data/models/poly/pbst

# -t pbst \