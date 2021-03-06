RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT=cuad_test_model_size
BASE_MODEL_NAME=bal_ckpt_modelsize_del_
echo "${PROJECT}"

# Run inf 
TO_DEL="9"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="10"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="11"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="5"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="6"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="2"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="1"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="4"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="3 6"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="9 10"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="11 3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="2 4 6"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="5 2 4"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="1 2"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="5 9"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="8 9"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="10 9"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2


# baseline
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --model_path deepset/roberta-base-squad2

TO_DEL="9 10 1"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="9 1 3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="10 1 3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="1 10"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="1 3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="10 9 4 11"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2

TO_DEL="10 9 3"
MODEL_POSTFIX=$(echo "${TO_DEL}" | tr " " _)
printf "${RED} Running script for deleting layer ${TO_DEL} ${NC} - And running inference\n"
python main.py --model_name "${BASE_MODEL_NAME}${MODEL_POSTFIX}" --gpus 4 --project_name "${PROJECT}" --batch_size 16 --num_train_epochs 8 --only_first_answer_in_features f --dataset_num_workers 8 --test_examples_chunk_size 10 --test_examples_workers 8 --resume_from_pl_checkpoint ./out/checkpoints/checkpoint-val_loss-name_dataset-name_CUAD_model-type_roberta_only-first-ans_False_doc-stride_256_dataset-type_bal_features_v2_noans_1_squad_epoch05-val_loss0.146-aupr0.000-aupr80rec0.000-hasans0.000-global_step1864.ckpt --test_model 1 --delete_transformer_layers $TO_DEL --model_path deepset/roberta-base-squad2