CUDA_VISIBLE_DEVICES=3 \
nohup python -u train/train_unetpp_gan.py \
    --train \
    --test \
    --predict \
    --early-stopping \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/UNetpp_GAN \
    --elevation-id 1 2 3 \
    --max-iterations 100000 \
    --augment-ratio 4 \
    --batch-size 32 \
    --num-threads 10 \
    --num-workers 10 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --weight-recon 100 \
    --case-indices 15451 \
    --case-anchor 160 \
    --case-blockage-len 40 \
    --display-interval 50 \
    > results/train_unetpp_gan.log 2>&1 &
