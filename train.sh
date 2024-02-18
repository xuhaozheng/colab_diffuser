export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="sintel"
export DATASET_DIR="/media/neuralmaster/9d5af100-a900-4e89-bab1-43c8b5025daf/neuromaster/Documents/haozheng/StereoDS/MPI-Sintel-stereo-training-20150305"

accelerate launch --mixed_precision="fp16" --multi_gpu  train_depth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_data_dir=$DATASET_DIR \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="depth-model"