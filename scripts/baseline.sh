nohup python -u baseline.py \
    --data-path /data/gaf/SBandBasicUnzip \
    --output-path results/Baseline \
    --test \
    --predict \
    --sample-index 15441 \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/baseline.log 2>&1 &