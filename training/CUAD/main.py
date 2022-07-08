# Run training of the model using pytorch lightning
from collections import OrderedDict
from lightning import PLQAModel
from models import QAModel
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, SquadV2Processor, squad_convert_examples_to_features
import argparse
import os
from data import get_balanced_dataset
from utils_v2 import get_balanced_dataset_v2
import logging
import gc
import sys
from utils import delete_encoding_layers
from helpers import str2bool, make_dataset_path, set_seed, make_dataset_name_base
from utils_valid import feature_path
import random
import os
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Model types confirmed to be working
MODEL_CLASSES = set(['roberta', 'deberta'])


def main(args):
    global hparams
    # run specific random int
    rand_v = random.randint(0, 10000)
    args.random_int = rand_v
    hparams = vars(args)
    set_seed(hparams)

    # Tokenizer and model
    logging.info("Loading Tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        do_lower_case=args.do_lower_case, use_fast=False)
    config = AutoConfig.from_pretrained(
        args.model_path,
        cache_dir=None,
    )
    robertaQA = AutoModelForQuestionAnswering.from_pretrained(
        args.model_path,
        config=config,
        cache_dir=None,
    )
    robertaQA.train()
    model = QAModel(hparams, robertaQA)

    # Checkpoint loading will fail if we delete layer due to mismatch between dicts
    if args.resume_from_pl_checkpoint and args.delete_transformer_layers:
        # load state dict
        state_dict = torch.load(
            args.resume_from_pl_checkpoint, map_location=torch.device('cpu'))
        state_dict = state_dict['state_dict']

        # Cleanup of naming used in the pl checkpoint #TODO: Less hardcoded
        state_dict = OrderedDict((k[6:] if 'model.' in k else k, v)
                                 for k, v in state_dict.items())

        model.load_state_dict(state_dict)

        # ensure it's not reloaded
        args.resume_from_pl_checkpoint = None

    # Delete layers if specified and add size
    size_1 = robertaQA.num_parameters()
    size_2 = -1
    if args.delete_transformer_layers:
        logging.info(
            f"Deleting transformer layers {args.delete_transformer_layers}")
        model.model = delete_encoding_layers(args, model.model)
        size_2 = model.model.num_parameters()
        logging.info(f"Deleted {size_1-size_2} parameters")
        # Percent decreased model size
        logging.info(
            f"New model size percentage of old : {int(100*size_2/size_1)}%")
        # Free up memory from deleted layers
        gc.collect()

    # Valid/test dataset
    logging.info("Loading val/test dataset")
    valid_dataset = get_or_create_dataset(
        args, tokenizer, evaluate=True)

    val_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataset_num_workers, drop_last=False)
    # Train dataset
    logging.info("Loading train dataset")
    dataset = get_or_create_dataset(
        args, tokenizer, evaluate=False)
    logging.info(f"Total dataset size Train: {len(dataset)}")

    # Balanced dataset logic and rank zero
    if args.dataset_balance_frac:
        logging.info("Using balanced dataset v2")
        train_dataset, q_count = get_balanced_dataset_v2(
            dataset, tokenizer, keep_frac=args.dataset_balance_frac, return_positives_dict=True)
        logging.info(f"Total annotations: {sum(q_count.values())}")
        logging.info(f"Question category annotations: {q_count}")
    else:
        logging.info("Using balanced dataset v1")
        train_dataset = get_balanced_dataset(dataset)

    logging.info(f"Dataset balanced size Train: {len(train_dataset)}")

    # Terminate if only create dataset
    if args.only_create_dataset:
        logging.info("Created dataset. Exiting")
        return

    del dataset
    gc.collect()
    logging.info(f"Batch size: {args.batch_size}")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataset_num_workers)
    hparams['train_set_size'] = len(train_dataset)
    hparams['val_set_size'] = len(valid_dataset)

    # Training preparation
    logging.info("Preparing training")

    litModel = PLQAModel(model, args, hparams, tokenizer)

    # Load pytorch lightning model
    if args.lit_model_path:
        litModel.model = torch.load(args.lit_model_path)
        logging.info(f"Loaded model from {args.lit_model_path}")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    _checkpoint_ending = 'epoch{epoch:02d}-val_loss{epoch_valid_loss:.3f}-hasans{performance_stats_valid_HasAns_f1:.3f}-global_step{step}'

    # Callback issues due to PL issue 11242 for other metrics than loss
    checkpoint_val_loss_callback = ModelCheckpoint(
        monitor='epoch_valid_loss',
        dirpath='out/checkpoints/',
        filename=f'checkpoint-val_loss-name_{make_dataset_name_base(args)}_{args.model_name}_' +
        _checkpoint_ending,
        auto_insert_metric_name=False,
        save_top_k=args.top_k_checkpoints,
        save_weights_only=True
    )

    wandb_logger = WandbLogger(
        project=args.project_name, entity="gustavhartz")

    gpus = args.gpus
    if args.specify_gpus:
        gpus = args.specify_gpus

    trainer = pl.Trainer(gpus=gpus, max_epochs=args.num_train_epochs,
                         logger=wandb_logger,
                         strategy='ddp',
                         callbacks=[lr_monitor, checkpoint_val_loss_callback],
                         auto_select_gpus=args.auto_select_gpus,
                         val_check_interval=args.val_check_interval)
    # if test model
    if args.test_model:
        del train_dataset
        del train_loader
        gc.collect()
        logging.info("Running test inference")
        trainer.test(litModel, val_loader,
                     ckpt_path=args.resume_from_pl_checkpoint if not args.delete_transformer_layers else None)
        dist.barrier()
        sys.exit(0)
    # Training
    logging.info("Starting training")
    trainer.fit(litModel, train_loader, val_loader,
                ckpt_path=args.resume_from_pl_checkpoint)
    logging.info("Training finished")


def build_and_cache_dataset(args, tokenizer, dataset_path, evaluate=False):
    processor = SquadV2Processor()
    load_data = os.path.exists(dataset_path+"_examples") and args.cached_data
    if load_data:
        logging.info("Using cached examples")
        examples = torch.load(dataset_path+"_examples")
        logging.info(f"Loaded {len(examples)} examples")
    # Use train processor on both train and valid for getting validation loss
    filename = args.predict_file if evaluate else args.train_file
    examples = processor.get_train_examples(
        None, filename=filename)
    if not load_data:
        logging.info(f"Saving {len(examples)} examples")
        torch.save(examples, dataset_path+"_examples")

    logging.info(
        "Creating features... This is a long running process and can take multiple hours")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=args.dataset_creation_threads,
        only_first_answer_in_features=args.only_first_answer_in_features,
    )
    # Assert that we are using the custom dataset with the feature indexes
    assert len(
        dataset[0]) == 9, "Dataset is not the correct size. Did you remember to use the customs squad.py file in transformers?"

    # Free up memory
    del examples
    gc.collect()
    logging.info(f"Saving {len(features)} features")
    # Save features for use in training validation
    if evaluate:
        # check if features folder exists in args.our_dir
        if not os.path.exists(os.path.join(args.out_dir, "features")):
            os.mkdir(os.path.join(args.out_dir, "features"))
        # dump features to file
        logging.info(
            f"Saving the individual features to {os.path.join(args.out_dir, 'features')}")
        for idx, feature in enumerate(features):
            torch.save(feature, feature_path(args, idx))
        logging.info(f"Saved the individual feature list files")
    logging.info("Saving features to cache file")
    torch.save(features, dataset_path+"_features")
    # Free up memory
    del features
    gc.collect()
    logging.info("Saving dataset to cache file")
    torch.save(dataset, dataset_path+"_dataset")
    # Print stuff saved
    logging.info(f"Saved dataset, features, and examples to: {dataset_path}")
    return dataset


def get_or_create_dataset(args, tokenizer, evaluate=False):
    dataset_path = make_dataset_path(args, evaluate)
    if args.cached_data and os.path.exists(dataset_path+"_dataset"):
        # Load dataset from cache if it exists
        dataset = torch.load(dataset_path+"_dataset")
    else:
        logging.info("Creating dataset")
        dataset = build_and_cache_dataset(
            args, tokenizer, dataset_path, evaluate=evaluate)
    return dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    # Argument for model type
    argparser.add_argument('--model_path', type=str,
                           default='Rakib/roberta-base-on-cuad', help='Model type to use')
    # Model name
    argparser.add_argument('--model_name', type=str,
                           default="roberta_cuad_checkpoint", help='Friendly name for the model. No special chars')
    # Model type
    argparser.add_argument('--model_type', type=str,
                           default='roberta', help='Model type to use')
    # Learning rate
    argparser.add_argument('--lr', type=float,
                           default=1e-4, help='Learning rate')
    # Batch size
    argparser.add_argument('--batch_size', type=int,
                           default=8, help='Batch size')
    # Number of labels
    argparser.add_argument('--num_labels', type=int,
                           default=2, help='Number of labels')
    # Hidden size
    argparser.add_argument('--hidden_size', type=int,
                           default=768, help='Hidden size')
    # Number of train epochs
    argparser.add_argument('--num_train_epochs', type=int,
                           default=5, help='Number of train epochs')
    # Log text every n batch
    argparser.add_argument('--log_text_every_n_batch',
                           type=int, default=30, help='Log text every n batch')
    # Log text every n batch
    argparser.add_argument('--log_text_every_n_batch_valid',
                           type=int, default=10, help='Log text every n batch')
    # Adam epsilon
    argparser.add_argument('--adam_epsilon', type=float,
                           default=1e-8, help='Adam epsilon')
    # Warmup steps
    argparser.add_argument('--warmup_steps', type=int,
                           default=100, help='Warmup steps')
    # Number of gpus
    argparser.add_argument('--gpus', type=int, default=4,
                           help='Number of gpus')
    # Seed
    argparser.add_argument('--seed', type=int, default=42, help='Seed')

    # Model max sequence length soft limit
    argparser.add_argument('--doc_stride', type=int,
                           default=256, help='document stride')
    # Create encodings argument - boolean
    argparser.add_argument('--max_seq_length', type=int,
                           default=512, help='Max sequence length')
    # Used cached data
    argparser.add_argument('--cached_data', type=str2bool, nargs='?',
                           const=True, default=True, help='Use cached data')
    # Train file
    argparser.add_argument('--train_file', type=str,
                           default='../../data/train_separate_questions.json', help='Train file')
    # Predict file
    argparser.add_argument('--predict_file', type=str,
                           default='../../data/test.json', help='Predict file')
    # Predict file type
    argparser.add_argument('--predict_file_version', type=str,
                           default='test', help='Predict file version. This is used to determine the output file name as we have different versions of the test file')
    # Train file type
    argparser.add_argument('--train_file_version', type=str,
                           default='train', help='Train file version. This is used to determine the output file name as we have different versions of the train file')
    # Out dir
    argparser.add_argument('--out_dir', type=str,
                           default='./out', help='Out dir')
    # model version
    argparser.add_argument('--model_version', type=str,
                           default='v1', help='Model version')
    # Do lower case
    argparser.add_argument('--do_lower_case', type=str2bool, nargs='?',
                           const=True, default=True, help='Do lower case')
    # max query length
    argparser.add_argument('--max_query_length', type=int,
                           default=64, help='Max query length')
    # Max answer length
    argparser.add_argument('--max_answer_length', type=int,
                           default=200, help='Max answer length')
    # Null score diff threshold
    argparser.add_argument('--null_score_diff_threshold', type=float,
                           default=0.0, help='Null score diff threshold')
    # Project name
    argparser.add_argument('--project_name', type=str,
                           default='roberta_cuad_checkpoint', help='Project name')
    # N best size AUPR
    argparser.add_argument('--n_best_size', type=int,
                           default=20, help='N best size')
    # N best size squad_evaluate
    argparser.add_argument('--n_best_size_squad_evaluate', type=int,
                           default=3, help='N best size for the second squad evaluation')
    # Dataset name
    argparser.add_argument('--dataset_name', type=str,
                           default='CUAD', help='Dataset name')
    # Dataset numworkers
    argparser.add_argument('--dataset_num_workers', type=int,
                           default=2, help='Dataset numworkers')
    # Dataset creation threads
    argparser.add_argument('--dataset_creation_threads', type=int,
                           default=60, help='Dataset creation threads')
    # Test model
    argparser.add_argument('--test_model', type=str2bool, nargs='?',
                           const=True, default=False, help='Test model. This will not train the model and only run a single evaluation on the predict file using the CUAD metrics')
    # Verbose
    argparser.add_argument('--verbose', type=bool,
                           default=False, help='Verbose')
    # Delete transformer layers option
    argparser.add_argument("--delete_transformer_layers", nargs='+',
                           help='Delete layers. Used like --delete_transformer_layers 9 10 11. ', type=int, default=[])
    # Autoselect gpus
    argparser.add_argument('--auto_select_gpus', type=str2bool, nargs='?',
                           const=True, default=False, help='Autoselect gpus in pytorch lightning')
    # Specify gpus
    argparser.add_argument("--specify_gpus", nargs='+',
                           help='Used if a specific device should be used in pl training. For using device 1 and 2 use: --specific_gpus 1 2', type=int, default=[])
    # Resume from checkpoint
    argparser.add_argument('--resume_from_pl_checkpoint', type=str,
                           default=None, help='Path to pytorch lightning checkpoint')
    # Pytorch model load
    argparser.add_argument('--lit_model_path', type=str,
                           default=None, help='Path to pytorch model')
    # Val check interval 0.5
    argparser.add_argument('--val_check_interval', type=float,
                           default=0.5, help='Val check interval. See pytorch lightning documentation for more info')
    # only_first_answer_examples
    argparser.add_argument('--only_first_answer_in_features', type=str2bool, nargs='?',
                           const=True, default=True, help='When creating examples only use the first answer for each question')
    # only_create_dataset
    argparser.add_argument('--only_create_dataset', type=str2bool, nargs='?',
                           const=True, default=False, help='Terminate after creating dataset')
    # test_examples_workers
    argparser.add_argument('--test_examples_workers', type=int,
                           default=4, help='In testing the number of workers to use for processing data')
    # test_examples_chunk_size
    argparser.add_argument('--test_examples_chunk_size', type=int,
                           default=4, help="In testing the chunk size to use for processing data")
    # Top k checkpoints
    argparser.add_argument('--top_k_checkpoints', type=int,
                           default=2, help="PL model checkpoint tok_k configuration on min val loss")
    # Dataset type defined by the frac
    argparser.add_argument('--dataset_balance_frac', type=float,
                           default=None, help="Use to trigger balanced dataset creation and define the frac empty datapoints to use per category. 1 is same, 2 is twice as many")
    # Prediction logic used. V2 is the new one discussed in the paper inspired by the Huggingface QA pipeline approach
    argparser.add_argument('--prediction_logic', type=str,
                           default='v1', help='Use to trigger prediction logic. v1 is the original logic, v2 is the logic used in the paper')
    # Working directory
    argparser.add_argument('--working_dir', type=str,
                           default=None, help='Set/Change working directory')

    args = argparser.parse_args()

    if args.working_dir is not None:
        cur_dir = os.getcwd()
        if args.working_dir != cur_dir:
            print("Current directory is {}. Changing to {}".format(
                cur_dir, args.working_dir))
            os.chdir(args.working_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logging.warning("You've set a doc stride which may be superior to the document length in some "
                        "examples. This could result in errors when building features from the examples. Please reduce the doc "
                        "stride or increase the maximum length to ensure the features are correctly built."
                        )
    if args.test_model and not args.predict_file:
        logging.warning(
            "You've set test_model to True but not provided a predict file.  Please provide a predict file or set test_model to False.")
        sys.exit(1)

    # Only support the model from a checkpoint if we delete layers. Very hardcoded for now.
    if args.delete_transformer_layers and args.resume_from_pl_checkpoint:
        logging.warning(
            "You've set delete_transformer_layers to True but also provided a resume_from_pl_checkpoint.  This only allows for loading the model state dict and is very hardcoded. Might not work")

    if args.model_type not in MODEL_CLASSES:
        raise ValueError(
            "Unsupported model type {}. Might work but it's not tested".format(args.model_type))

    main(args)
