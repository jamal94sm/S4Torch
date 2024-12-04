#!/bin/bash
#SBATCH --account=def-arashmoh
#SBATCH --gpus-per-node=1
#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-03:00
#SBATCH --mail-user=<jamal73sm@gmail.com>
#SBATCH --mail-type=ALL

cd /home/shahab33/projects/def-arashmoh/shahab33
module purge
module load python
source ~/Mamba/bin/activate

python S4Torch/train.py --dataset=smnist --batch_size=16 --max_epochs=10 --lr=1e-2 --n_blocks=6 --d_model=128 --norm_type=layer 