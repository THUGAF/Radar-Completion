CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/GLCIC \
    --train \
    --test \
    --predict \
    --model GLCIC \
    --elevation-id 1 2 3 \
    --early-stopping \
    --batch-size 8 \
    --num-threads 8 \
    --num-workers 8 \
    --learning-rate 1e-4 \
    --weight-decay 1e-6 \
    --sample-index 14403 \
    --sample-anchor 180 \
    --sample-blockage-len 40 \
    --display-interval 50 \
    > results/GLCIC.log 2>&1 &
