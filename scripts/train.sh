CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/DilatedUNet \
    --test \
    --predict \
    --model DilatedUNet \
    --sample-index 15441 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/DilatedUNet.log 2>&1 &