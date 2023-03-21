nohup python -u train/test_baseline.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/Upper \
    --test \
    --predict \
    --elevation-id 1 2 3 \
    --batch-size 16 \
    --num-threads 1 \
    --num-workers 1 \
    --case-indices 15451 \
    --case-anchor 160 \
    --case-blockage-len 40 \
    --display-interval 50 \
    > results/test_baseline.log 2>&1 &