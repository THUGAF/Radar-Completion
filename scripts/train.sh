CUDA_VISIBLE_DEVICES=3 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results \
    --train \
    --test \
    --predict \
    --sample-index 18300 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > GLCIC.log 2>&1 &