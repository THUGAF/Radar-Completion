CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results \
    --train \
    --test \
    --predict \
    --sample-index 18300 \
    --early-stopping \
    --batch-size 8 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > GLCIC.log 2>&1 &