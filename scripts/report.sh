SPLIT=test
NAME=qmsum_quscore
ALPHA=0.6

for RUN in $(seq 1 1)
do
echo "************************************************************"
echo ${NAME}_${RUN}_${ALPHA}
python ../rouge/report_rouge.py \
--ref-path ../data/${SPLIT}.target \
--pred-paths \
    output/${NAME}_${RUN}_${ALPHA}/selected_checkpoint/predictions.${SPLIT}
done
