CUDA_VISIBLE_DEVICES=0 \
nohup python -u train.py \
    --data-path /data/gaf/SBandCRUnzip \
    --output-path results/ConvLSTM \
    --model EncoderForecaster \
    --train \
    --test \
    --predict \
    --sample-index 16840 \
    --max-iterations 50000 \
    --early-stopping \
    --batch-size 16 \
    --num-threads 8 \
    --num-workers 8 \
    --display-interval 20 \
    > ConvLSTM.log 2>&1 &
