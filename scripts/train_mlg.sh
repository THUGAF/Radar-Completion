CUDA_VISIBLE_DEVICES=0 \
nohup python -u train/train_mlg.py \
    --test \
    --predict \
    --early-stopping \
    --data-path /data/gaf/SBandRawNPZ \
    --output-path results/MLG \
    --elevation-id 1 2 3 4 5 6 7 8 \
    --max-iterations 100000 \
    --batch-size 32 \
    --num-threads 8 \
    --num-workers 8 \
    --learning-rate 1e-3 \
    --case-indices 13500 15451 \
    --case-anchor 315 40 \
    --case-blockage-len 40 40 \
    --display-interval 10 \
    > results/train_mlg.log 2>&1 &
