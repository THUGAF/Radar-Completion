CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/SmaAt_UNet \
    --model SmaAt_UNet \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --max-iterations 50000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > SmaAt_UNet.log 2>&1 &
