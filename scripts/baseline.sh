nohup python -u baseline.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/Upper \
    --test \
    --predict \
    --elevation-id 1 2 3 \
    --sample-index 14403 \
    --sample-anchor 180 \
    --sample-blockage-len 40 \
    --batch-size 8 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/upper.log 2>&1 &