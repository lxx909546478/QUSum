NAME=qmsum_quscore
SPLIT=test
NUM_RUNS=5
START=1
CUDA=1
ALPHA=0.6

for RUN in $(seq $START $NUM_RUNS)
do
  OUTPUT_DIR=output/${NAME}_${RUN}_${ALPHA}
  CUDA_VISIBLE_DEVICES=$CUDA python -u train.py \
    --test_file data/qmsum/preprocessed/${SPLIT}.jsonl \
    --do_predict \
    --fp16 \
    --model_name_or_path $OUTPUT_DIR/selected_checkpoint \
    --output_dir ${OUTPUT_DIR}/selected_checkpoint/predition_logs_${SPLIT} \
    --prediction_path ${OUTPUT_DIR}/selected_checkpoint/predictions.${SPLIT} \
    --max_source_length 512 \
    --generation_max_len 256 \
    --val_max_target_length 256 \
    --overwrite_output_dir \
    --per_device_eval_batch_size 1 \
    --multiencoder_type bart \
    --multiencoder_max_num_chunks 32 \
    --multiencoder_stride \
    --alpha $ALPHA \
    --predict_with_generate
done