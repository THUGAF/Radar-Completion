CUDA_VISIBLE_DEVICES=1 \
nohup python -u train/train_linear_regression.py \
    --test \
    --predict \
    --early-stopping \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/MLG \
    --elevation-id 1 2 3 4 5 6 7 8 \
    --max-iterations 100000 \
    --batch-size 32 \
    --num-threads 8 \
    --num-workers 8 \
    --learning-rate 1e-3 \
    --case-indices 13500 15451 \
    --case-anchor 270 40 \
    --case-blockage-len 40 40 \
    --display-interval 10 \
    > results/train_linear_regression.log 2>&1 &
