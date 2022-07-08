#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J Python_Test
### -- ask for number of cores (default: 1) -- 
#BSUB -n 32 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=5GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 8GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s174315@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err 

# here follow the commands you want to execute
module load python3/3.8.11
python3.8 -m pip install --user transformers==4.18.0 pytorch-lightning==1.6.0 pandas==1.4.2 wandb==0.12.14

WANDB_API_KEY=$YOUR_API_KEY

# Create the dataset
python3.8 /zhome/35/f/127154/transformers-legal-tasks/training/CUAD/main.py --only_first_answer_in_features f --test_examples_chunk_size 10 --only_create_dataset t --dataset_creation_threads 28 --train_file /zhome/35/f/127154/transformers-legal-tasks/data/train.json --train_file_version train_non_sep --working_dir /zhome/35/f/127154/transformers-legal-tasks/training/CUAD/

# Train the model 
# python3.8 /zhome/35/f/127154/transformers-legal-tasks/training/CUAD/main.py --model_path deepset/roberta-base-squad2 --model_name bal_features_v2_noans_1 --gpus 2 --project_name cuad_v3 --batch_size 16 --num_train_epochs 6 --only_first_answer_in_features f --dataset_num_workers 8 --val_check_interval 0.5 --test_examples_chunk_size 10 --test_examples_workers 8 --top_k_checkpoints 2 --dataset_balance_frac 1 --train_file /zhome/35/f/127154/transformers-legal-tasks/data/train.json --train_file_version train_non_sep