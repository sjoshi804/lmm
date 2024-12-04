WD="/home/sjoshi/lmm/lmm_synthetic/lm_train"
RUN_ID=wikitext
export WANDB_PROJECT="gptjpretrain"
deepspeed \
    --include localhost:3,4,5,6,7 \
    --master_port 29501 \
    $WD/pretrain_lm.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --vocab_size 50257 \
    --n_positions 1024 \
    --n_embd 768 \
    --n_layer 12 \
    --n_head 12 \
    --intermediate_size 3072 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --max_length 1024 \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --logging_dir $WD/logs \
        --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 1
