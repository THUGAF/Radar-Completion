nohup python -u train/test_bilinear.py \
    --data-path /data/gaf/SBandRawNPZ \
    --output-path results/Bilinear \
    --test \
    --predict \
    --elevation-id 1 2 3 \
    --batch-size 32 \
    --case-indices 13500 15451 \
    --case-anchor 315 40 \
    --case-blockage-len 40 40 \
    --display-interval 10 \
    > results/test_bilinear.log 2>&1 &