#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --account=def-hemmati-ab
#SBATCH --job-name=smac4vec-beluga
#SBATCH --mail-user=mohammadamin.abedi@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --cpus-per-task=40
#SBATCH --mem=186GB
module restore modules_ok
module load java
wd="/home/aminabd/scratch/code2vec"
source "$wd/env/bin/activate"
bash optimize.sh
