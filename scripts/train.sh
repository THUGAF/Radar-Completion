CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/GLCIC \
    --test \
    --predict \
    --model GLCIC \
    --sample-index 15441 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 50 \
    > results/GLCIC.log 2>&1 &
