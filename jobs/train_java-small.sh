#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --account=def-hemmati-ab
#SBATCH --job-name=train-code2vec-java-small
#SBATCH --mail-user=mohammadamin.abedi@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
module restore modules_ok
module load java
wd="/home/aminabd/scratch/code2vec"
source "$wd/env/bin/activate"
bash train.sh
