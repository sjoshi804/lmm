WD="/home/sjoshi/lmm/lmm_synthetic/mm_train"
RUN_ID=task_1_rq_1
export WANDB_PROJECT="vlm_training"

deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29501 \
    $WD/train_vlm.py \
    --deepspeed $WD/ds_configs/zero3.json \
    --data_path /home/sjoshi/lmm/lmm_synthetic/data/generated/v3.1_spatial_grid_multimodal \
    --split train \
    --gptj_model_path /home/sjoshi/lmm/lmm_synthetic/lm_train/checkpoints/sft_v3_spatial_grid/checkpoint-1955/ \
    --vision_encoder clip \
    --multimodal_projector linear \
    --freeze_lm False \
    --output_dir $WD/checkpoints/$RUN_ID \
    --per_device_train_batch_size 32 \
    --lr_scale_lm 2e-3 \
    --num_train_epochs 5 \
    --learning_rate 5e-3 \
    --logging_dir $WD/logs \
    --logging_steps 2 \
    --save_strategy epoch \
    --warmup_steps 25 \
    --save_total_limit 5 