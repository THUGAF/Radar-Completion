CUDA_VISIBLE_DEVICES=0 \
nohup python -u train/train_dsa_unet.py \
    --train \
    --test \
    --predict \
    --early-stopping \
    --data-path /data/gaf/SBandBasicPt \
    --output-path results/DSA_UNet \
    --elevation-id 1 2 3 \
    --max-iterations 100000 \
    --augment-ratio 4 \
    --batch-size 32 \
    --num-threads 8 \
    --num-workers 8 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --weight-recon 100 \
    --case-indices 15451 \
    --case-anchor 160 \
    --case-blockage-len 40 \
    --display-interval 50 \
    > results/train_dsa_unet.log 2>&1 &
