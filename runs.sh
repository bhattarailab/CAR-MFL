#!/bin/bash --login

#SBATCH --account yshresth_ai_management
#SBATCH --job-name img6Zero500
#SBATCH --output /scratch/ppoudel/img6Zero500%A_%a.out

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 32G
#SBATCH --time 3-00:00:00
#SBATCH --array=0-2

module load gcc miniconda3
conda activate fedml
wandb enabled

SEEDS=(11039 10059 4644)

CUDA_VISIBLE_DEVICES=0 python main.py --name img6Zero500  --algorithm fedavgln  --exp_dir /scratch/ppoudel/MICCAI2024/pub_study4/img6Zero500/ckpt_${SEEDS[$SLURM_ARRAY_TASK_ID]} --seed ${SEEDS[$SLURM_ARRAY_TASK_ID]} --num_clients 4 --img_clients 6 --txt_clients 0 --alpha 0.3 --server_config_path configs/fedavgin_server.yaml --client_config_path configs/client_configs.yaml --use_refinement