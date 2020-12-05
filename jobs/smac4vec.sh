#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:v100l:2
#SBATCH --account=def-hemmati-ab
#SBATCH --job-name=smac-code2vec-java-small
#SBATCH --mail-user=mohammadamin.abedi@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#SBATCH --cpus-per-task=32
#SBATCH --mem=187GB
module restore modules_ok
module load java
wd="/home/aminabd/scratch/code2vec"
source "$wd/env/bin/activate"
bash optimize.sh
