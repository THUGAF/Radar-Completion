CUDA_VISIBLE_DEVICES=2 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/DialatedUNet \
    --train \
    --test \
    --predict \
    --model DialatedUNet \
    --sample-index 15441 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/DialatedUNet.log 2>&1 &
