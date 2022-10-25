CUDA_VISIBLE_DEVICES=3 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicUnzip \
    --output-path results \
    --train \
    --test \
    --predict \
    --sample-index 16500 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/train.log 2>&1 &