CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet_SVRE \
    --model AttnUNet \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --max-iterations 50000 \
    --early-stopping \
    --batch-size 16 \
    --var-reg 0.1 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > AttnUNet_SVRE.log 2>&1 &
