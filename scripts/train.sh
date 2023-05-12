NAME=qmsum_quscore
START=1
NUM_RUNS=5
CUDA=1
ALPHA=0.6

for RUN in $(seq $START $NUM_RUNS)
do
  CUDA_VISIBLE_DEVICES=$CUDA python -u train.py \
  --train_file data/qmsum/preprocessed/train.jsonl \
  --validation_file data/qmsum/preprocessed/val.jsonl \
  --do_train \
  --do_eval \
  --learning_rate 5e-6 \
  --gradient_checkpointing \
  --model_name_or_path facebook/bart-large \
  --metric_for_best_model eval_mean_rouge \
  --output_dir output/${NAME}_${RUN}_${ALPHA} \
  --per_device_train_batch_size 1 \
  --max_source_length 512 \
  --generation_max_len 256 \
  --val_max_target_length 256 \
  --overwrite_output_dir \
  --per_device_eval_batch_size 1 \
  --multiencoder_type bart \
  --multiencoder_max_num_chunks 32 \
  --multiencoder_stride \
  --predict_with_generate \
  --evaluation_strategy epoch \
  --num_train_epochs 10 \
  --save_strategy epoch \
  --logging_strategy epoch \
  --load_best_model_at_end \
  --compute_rouge_for_train True \
  --alpha $ALPHA \
  --seed $RUN &> ${NAME}_${RUN}_${ALPHA}.out
done
#  --metric_for_best_model rouge1_plus_rouge2 \
