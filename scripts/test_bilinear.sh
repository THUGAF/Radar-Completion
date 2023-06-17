nohup python -u train/test_bilinear.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/Bilinear \
    --test \
    --predict \
    --elevation-id 1 2 3 \
    --batch-size 32 \
    --case-indices 15451 \
    --case-anchor 160 \
    --case-blockage-len 40 \
    --display-interval 10 \
    > results/test_bilinear.log 2>&1 &