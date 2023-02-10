CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/UNetppL3 \
    --train \
    --test \
    --predict \
    --model UNetppL3 \
    --elevation-id 1 2 3 \
    --sample-index 14403 \
    --sample-anchor 180 \
    --sample-blockage-len 40 \
    --early-stopping \
    --batch-size 8 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/UNetppL3.log 2>&1 &
