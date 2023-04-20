#!/bin/env bash

#SBATCH -A SNIC2022-5-184      # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-12:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A100:4  # choosing no. GPUs and their type
#SBATCH -J DBPP-with-fade           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/mimer/NOBACKUP/groups/snic2021-7-127/enliden/databasesamplerv2/mmdetection3d/singularity/mmdetection3d_with_ds_choice.sif
DIR=pointpillars
WORK=work_dirs_pp_with_fade
CONFIG=hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_fade_resume
# Print the PyTorch version then exit
# singularity exec $CONTAINER python -c "import torch; print(torch.__version__)"
cd /mimer/NOBACKUP/groups/snic2021-7-127/enliden/databasesamplerv2/mmdetection3d/
singularity exec $CONTAINER bash tools/dist_train.sh configs/$DIR/$CONFIG.py 4 --work-dir $WORK
