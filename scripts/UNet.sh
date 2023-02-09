CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/UNet \
    --train \
    --test \
    --predict \
    --model UNet \
    --elevation-id 1 2 \
    --sample-index 14403 \
    --sample-anchor 180 \
    --sample-blockage-len 40 \
    --early-stopping \
    --batch-size 8 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/UNet.log 2>&1 &
