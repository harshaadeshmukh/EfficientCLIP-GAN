# Set the path to the configuration file
cfg=$1
# Define the batch size for training
batch_size=32
# Specify the path to the pretrained model
pretrained_model='./path/to/model.pth'
# Indicate whether to use multiple GPUs for training
multi_gpus=True
# Specify whether to use mixed precision training
mixed_precision=True

# Set the number of nodes for distributed training
nodes=1
# Specify the number of worker processes for data loading
num_workers=4
# Set the port number for the master process
master_port=11277
# Construct a string to represent the training setup
stamp=gpu${nodes}MP_${mixed_precision}

# Restrict CUDA to use only the specified GPU device (device index 0)
CUDA_VISIBLE_DEVICES=0 \
# Launch the PyTorch distributed training script
torchrun \
    --nproc_per_node=$nodes \
    --master_port=$master_port \
    src/test.py \
    # Pass the training stamp as an argument to the training script
    --stamp $stamp \
    # Pass the configuration file path as an argument to the training script
    --cfg $cfg \
    # Pass the mixed precision flag to the training script
    --mixed_precision $mixed_precision \
    # Pass the batch size to the training script
    --batch_size $batch_size \
    # Pass the number of data loader workers to the training script
    --num_workers $num_workers \
    # Pass the flag indicating whether to use multiple GPUs to the training script
    --multi_gpus $multi_gpus \
    # Pass the path to the pretrained model to the training script
    --pretrained_model_path $pretrained_model
