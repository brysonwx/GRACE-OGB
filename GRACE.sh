#!/bin/bash
#SBATCH --job-name=GRACE-OGB-ARXIV 
#SBATCH --output=GRACE-OUTPUT.log       
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --partition=test-hpc-1
#SBATCH --gres=gpu:1
echo "Hello World! This is my GRACE-OGB-ARXIV job on Slurm."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES" $CUDA_VISIBLE_DEVICES
source /home/<username>/anaconda3/bin/activate GRACE-OGB
which python
cd /home/<username>/projects/GRACE-OGB
python train.py --gpu_id 0 --dataset ogbn-arxiv
echo "Job completed successfully."