REPO=$PWD
GPU=${1:-0}
MODEL=${2:-bert-base-multilingual-cased}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU

TASK='xnli'
LR=2e-5
EPOCH=15
MAXL=128
TRAIN_LANG="en"
LANGS="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh"
BPE_DROP=0.2
KL=0.6 

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
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

for SEED in 1;
do
SAVE_DIR="$OUT_DIR/$TASK/mvr-${MODEL}-epoch${EPOCH}/"
mkdir -p $SAVE_DIR

python $PWD/third_party/llm-analysis.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --train_language $TRAIN_LANG \
  --task_name $TASK \
  --data_dir $DATA_DIR/${TASK} \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --output_dir $SAVE_DIR/ \
  --predict_languages $LANGS \
  --tokenizer_name "$SAVE_DIR/checkpoint-best/"
done
