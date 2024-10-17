CHECKPOINT_PATH="./checkpoints"

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$CHECKPOINT_PATH/10_14_2024_ai2d_gen_cap" \
    --tasks ai2d \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$CHECKPOINT_PATH/10_14_2024_spatial_map_gen_cap" \
    --tasks spatial_map \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$CHECKPOINT_PATH/10_14_2024_spatial_map_mminstruct" \
    --tasks spatial_map \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="$CHECKPOINT_PATH/10_14_2024_spatial_map_skyline" \
    --tasks spatial_map \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/