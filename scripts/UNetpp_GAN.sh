CUDA_VISIBLE_DEVICES=1 \
nohup python -u train.py \
    --train \
    --test \
    --predict \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/UNetpp_GAN \
    --model UNetpp_GAN \
    --elevation-id 1 2 3 \
    --max-iterations 100000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --learning-rate 1e-4 \
    --beta1 0.5 \
    --weight-decay 1e-6 \
    --weight-recon 10 \
    --sample-index 14403 \
    --sample-anchor 180 \
    --sample-blockage-len 40 \
    --display-interval 50 \
    > results/UNetpp_GAN.log 2>&1 &
