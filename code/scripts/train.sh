# Assigning the value of the first command line argument to the variable 'cfg'
cfg=$1
# Setting the batch size to 64
batch_size=32

# Setting the starting epoch number to 1
state_epoch=1
# Path to the pretrained model file
pretrained_model_path='./path/to/pretrained_model.pth'
# Directory for logging
log_dir='new'

# whether to use multiple GPUs for training
multi_gpus=True
# whether to use mixed precision training
mixed_precision=True

# Number of nodes (machines) for distributed training
nodes=1
# Number of worker processes for data loading
num_workers=4
# Port number for communication between master and worker processes
master_port=11266
# Stamp used for identifying the current training setup
stamp=gpu${nodes}MP_${mixed_precision}

# Command for launching distributed training using the specified number of nodes and GPUs
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$nodes --master_port=$master_port src/train.py \
                    # Passing arguments to the training script
                    --stamp $stamp \  # Stamp indicating the current training setup
                    --cfg $cfg \  # Path to the configuration file
                    --mixed_precision $mixed_precision \  # Flag indicating whether to use mixed precision
                    --log_dir $log_dir \  # Directory for logging
                    --batch_size $batch_size \  # Batch size for training
                    --state_epoch $state_epoch \  # Starting epoch number
                    --num_workers $num_workers \  # Number of worker processes for data loading
                    --multi_gpus $multi_gpus \  # Flag indicating whether to use multiple GPUs
                    --pretrained_model_path $pretrained_model_path \  # Path to the pretrained model file
