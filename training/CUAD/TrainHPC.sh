#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J CUAD_Training
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 5GB of memory per core/slot --
#BSUB -R "rusage[mem=5GB]"
### -- specify that we want the job to get killed if it exceeds 8 GB per core/slot --
#BSUB -M 8GB
### -- set walltime limit: hh:mm --
#BSUB -W 12:00
### GPU Settings
### -- Select the resources: 2 gpus in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"

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
#BSUB -o output_%J_cuad.out
#BSUB -e error_%J_cuad.err

# GPU prep
nvidia-smi

# here follow the commands you want to execute
module load python3/3.9.11

pip3 install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install --user transformers pytorch-lightning==1.6.0 pandas wandb sklearn

# Add the updated transformer file - hardcoded for now
cp /zhome/35/f/127154/transformers-legal-tasks/squad.py /zhome/35/f/127154/.local/lib/python3.9/site-packages/transformers/data/processors/squad.py

# Train the model
python3 /zhome/35/f/127154/transformers-legal-tasks/training/CUAD/main.py --only_first_answer_in_features f --test_examples_chunk_size 10 --train_file /zhome/35/f/127154/transformers-legal-tasks/data/train.json --train_file_version train_non_sep --working_dir /zhome/35/f/127154/transformers-legal-tasks/training/CUAD --out_dir /work3/s174315/out/ --model_path deepset/roberta-base-squad2 --model_name cuad_batch_32 --gpus 2 --project_name cuad_hpc --batch_size 32 --num_train_epochs 6 --dataset_num_workers 6 --val_check_interval 0.5 --test_examples_workers 6 --top_k_checkpoints 2 --dataset_balance_frac 1 --label_smoothing 0.0