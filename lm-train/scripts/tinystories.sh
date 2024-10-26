WD="/home/sjoshi/lmm/lm-train"
RUN_ID=tinystories
export WANDB_PROJECT="gptjpretrain"
deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    --master_port 29501 \
    $WD/pretrain_gptj.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --vocab_size 50257 \
    --n_positions 1024 \
    --n_embd 768 \
    --n_layer 12 \
    --n_head 12 \
    --intermediate_size 3072 \
    --dataset_name roneneldan/TinyStories \
    --max_length 1024 \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --logging_dir $WD/logs \
        --logging_steps 100 \
    --save_strategy epoch \
    --save_total_limit 2
