#!/bin/bash
#SBATCH --job-name=eval_job_yt
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --nodelist=bumblebee.ib
#SBATCH --gpus=4 
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm/eval/%j.out

#with lora
python llava/eval/infer_yt.py \
    --model_base lmms-lab/LLaVA-Video-72B-Qwen2 \
    --model_path work_dirs/llava_video_7B_64f_yt \
    --data_path DATA/eval_yt.json \
    --results_dir results/yt_scam/ \
    --use_time_ins \
    --max_frames_num 64