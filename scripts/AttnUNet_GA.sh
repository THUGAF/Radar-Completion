CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/AttnUNet_GA \
    --model AttnUNet \
    --add-gan \
    --train \
    --test \
    --predict \
    --ensemble-members 3 \
    --sample-index 16840 \
    --max-iterations 50000 \
    --early-stopping \
    --gan-reg 0.1 \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > AttnUNet_GA.log 2>&1 &
