#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-hemmati-ab
#SBATCH --job-name=predict-code2vec
#SBATCH --mail-user=mohammadamin.abedi@ucalgary.ca
#SBATCH --mail-type=END
#SBATCH --mem=125G
module restore modules_ok
module load java
wd="/home/aminabd/scratch/code2vec"
source "$wd/env/bin/activate"
echo "$wd"
python "$wd/code2vec.py" --load $wd/../models/java14_model/saved_model_iter8.release --predict
