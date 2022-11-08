CUDA_VISIBLE_DEVICES=3 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/UNet \
    --train \
    --test \
    --predict \
    --model UNet \
    --sample-index 15441 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/UNet.log 2>&1 &