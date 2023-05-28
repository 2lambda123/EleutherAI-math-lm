#!/bin/bash
#SBATCH --job-name="mathlm"
#SBATCH --partition=g40
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=6           # Number of cores per tasks
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=1.4b_12000_early_%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=1.4b_12000_early_%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --comment=eleuther
#SBATCH --time=2:00:00

# script designed for the eleuther.hpc.stability.ai cluster

# setup the environment using the script we created before
source /fsx/hailey/conda_setup.sh
#source /fsx/quentin/setup.sh

ds_report

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json
export TORCH_EXTENSIONS_DIR=./extensions/

# Move to the gpt-neox install
TRAIN_PATH=/fsx/mathlm0/gpt-neox/
cd $TRAIN_PATH

# Write the hostfile for this job
export MASTER_ADDR=$(echo $MASTER_ADDR | cut -d '-' -f 2- | tr '-' '.')
bash /fsx/hailey/write_ip_hostfile.sh
export DLTS_HOSTFILE=/fsx/hailey/hostfiles/hosts_$SLURM_JOBID

export WANDB_API_KEY=07cebf97416da7fa921b74774ef771f52d4e49e9
wandb login

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
        --conf_dir /fsx/mathlm0/gpt-neox/configs/v0.4/ 1.4b-12000-early.yml
