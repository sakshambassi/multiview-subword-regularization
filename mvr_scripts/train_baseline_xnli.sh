#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
LR=2e-5
EPOCH=15
MAXL=128
TRAIN_LANG="en"
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
  LR=3e-5
else
  BATCH_SIZE=128
  GRAD_ACC=4
  LR=2e-5
fi

for SEED in 64;
do
SAVE_DIR="$OUT_DIR/$TASK/baseline-${MODEL}-LR${LR}-epoch${EPOCH}-MaxLen${MAXL}_train${TRAIN_LANG}_s${SEED}/"
mkdir -p $SAVE_DIR

python $PWD/third_party/run_baseline_classify.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language $TRAIN_LANG \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir $DATA_DIR/${TASK} \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --save_steps 1000 \
  --eval_all_checkpoints \
  --log_file 'train' \
  --predict_languages $LANGS \
  --save_only_best_checkpoint \
  --overwrite_output_dir \
  --seed $SEED \
  --eval_test_set $LC
done
