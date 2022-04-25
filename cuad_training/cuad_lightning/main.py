# Run training of the model using pytorch lightning
from lightning import PLQAModel
from models import QAModel
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering, SquadV2Processor, squad_convert_examples_to_features
import random
import numpy as np
import argparse
import os
from data import get_balanced_dataset
import logging
import gc
import sys
import string
from utils import delete_encoding_layers
from utils_valid import feature_path
logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')



def set_seed(args):
    """Create a random seed for reproducibility

    Args:
        args (_type_): Argparse
    """    
    seed = args.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.get('gpus', 0) > 0:
        torch.cuda.manual_seed_all(seed)


def main(args):
    global hparams
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
    # Delete layers if specified and add size
    size_1 = robertaQA.num_parameters()
    size_2 = -1
    if args.delete_transformer_layers:
        logging.info(f"Deleting transformer layers {args.delete_transformer_layers}")
        robertaQA = delete_encoding_layers(args, robertaQA)
        size_2 = robertaQA.num_parameters()
        logging.info(f"Deleted {size_1-size_2} parameters")
        # Percent decreased model size
        logging.info(f"New model size percentage of old : {int(100*size_2/size_1)}%")
        # Free up memory from deleted layers
        gc.collect()
    
    hparams['model_params'] = size_1 if not args.delete_transformer_layers else size_2
    hparams['deleted_layers'] = "" if not args.delete_transformer_layers else str(args.delete_transformer_layers)
    robertaQA.train()
    model = QAModel(hparams, robertaQA)


    # Valid/test dataset
    logging.info("Loading val/test dataset")
    valid_dataset = get_or_create_dataset(
        args, tokenizer, evaluate=True)
    
    val_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataset_num_workers)
    # Train dataset
    logging.info("Loading train dataset")
    dataset = get_or_create_dataset(
        args, tokenizer, evaluate=False)
    logging.info(f"Total dataset size Train: {len(dataset)}")
    train_dataset = get_balanced_dataset(dataset)
    logging.info(f"Dataset balanced size Train: {len(train_dataset)}")
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

    wandb_logger = WandbLogger(
        project=args.project_name, entity="gustavhartz")

    gpus=args.gpus
    if args.specify_gpus:
        gpus=args.specify_gpus

    trainer = pl.Trainer(gpus=gpus, max_epochs=args.num_train_epochs,
                         logger=wandb_logger, 
                         strategy='ddp', 
                         callbacks=[lr_monitor], 
                         auto_select_gpus=args.auto_select_gpus)
    # if test model
    if args.test_model:
        del train_dataset
        del train_loader
        gc.collect()
        logging.info("Running test inference")
        trainer.validate(litModel,val_loader, ckpt_path=args.resume_from_pl_checkpoint)
        sys.exit(0)
    # Training
    logging.info("Starting training")
    trainer.fit(litModel, train_loader, val_loader, ckpt_path=args.resume_from_pl_checkpoint)
    logging.info("Training finished")
    logging.info("Saving model")
    torch.save(litModel.model,
               f"./{args.model_name}_{args.model_type}_{args.model_version}_{random_string(5)}_model.pt")


# Function that generates random string of length n
def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))


def build_and_cache_dataset(args, tokenizer, dataset_path, evaluate=False):
    processor = SquadV2Processor()
    load_data = os.path.exists(dataset_path+"_examples") and args.cached_data
    if load_data:
        logging.info("Using cached examples")
        examples = torch.load(dataset_path+"_examples")
    # Use train processor on both train and valid for getting validation loss
    filename = args.predict_file if evaluate else args.train_file
    examples = processor.get_train_examples(
            None, filename=filename)
    if not load_data:
        logging.info("Saving examples...")
        torch.save(examples, dataset_path+"_examples")
    

    logging.info("Creating features... This is a long running process and can take multiple hours")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=args.dataset_creation_threads
        )
    print(f"Dataset size {len(dataset)}")

    # Assert that we are using the custom dataset with the feature indexes
    try:
        assert len(dataset) == 9
    except:
        raise Exception("Dataset is not the correct size. Did you remember to use the customs squad.py file in transformers?")


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
        logging.info(f"Saving the individual features to {os.path.join(args.out_dir, 'features')}")
        for idx, feature in enumerate(features):
            torch.save(feature, feature_path(args,idx))
        logging.info(f"Saved the individual feature list files")
    logging.info("Saving features to cache file")
    torch.save(features, dataset_path+"_features")
    # Free up memory
    del features
    gc.collect()
    logging.info("Saving dataset to cache file")
    torch.save(dataset, dataset_path+"_dataset")
    #Print stuff saved
    logging.info(f"Saved dataset, features, and examples to: {dataset_path}")
    return dataset


def get_or_create_dataset(args, tokenizer, evaluate=False):
    dataset = None
    DATASET_NAME = args.dataset_name+"_"+args.model_type
    dataset_path = os.path.join(
        args.out_dir, DATASET_NAME + "_eval_" + args.predict_file_version if evaluate else DATASET_NAME + "_train_" + args.train_file_version)
    if args.cached_data and os.path.exists(dataset_path+"_dataset"):
        # Load dataset from cache if it exists
        dataset= torch.load(dataset_path+"_dataset")
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
    # Number of workers
    argparser.add_argument('--num_workers', type=int,
                           default=5, help='Number of workers')
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
    argparser.add_argument('--cached_data', type=bool,
                           default=True, help='Use cached data')
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
    argparser.add_argument('--do_lower_case', type=bool,
                           default=True, help='Do lower case')
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
    # N best size
    argparser.add_argument('--n_best_size', type=int,
                            default=1, help='N best size')
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
    argparser.add_argument('--test_model', type=bool,
                            default=False, help='Test model. This will not train the model and only run a single evaluation on the predict file using the CUAD metrics')
    # Verbose
    argparser.add_argument('--verbose', type=bool,
                            default=True, help='Verbose')
    # Delete transformer layers option
    argparser.add_argument("--delete_transformer_layers", nargs='+',
                            help='Delete layers. Used like --delete_transformer_layers 9 10 11. ', type=int, default=[])
    # Autoselect gpus
    argparser.add_argument('--auto_select_gpus', type=bool,
                            default=False, help='Autoselect gpus in pytorch lightning')
    # Specify gpus
    argparser.add_argument("--specify_gpus", nargs='+',
                           help='Used if a specific device should be used in pl training. For using device 1 and 2 use: --specific_gpus 1 2', type=int, default=[])
    # Resume from checkpoint
    argparser.add_argument('--resume_from_pl_checkpoint', type=str, 
                            default=None, help='Path to pytorch lightning checkpoint')
    # Pytorch model load
    argparser.add_argument('--lit_model_path', type=str,
                            default=None, help='Path to pytorch model')

    args = argparser.parse_args()
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logging.warning("You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )
    if args.test_model and not args.predict_file:
        logging.warning("You've set test_model to True but not provided a predict file.  Please provide a predict file or set test_model to False.")
        sys.exit(1)
    if args.test_model and ((args.gpus > 1) or (len(args.specify_gpus)>1)):
        logging.warning("You've set test_model to True but you have more than one GPU.  Testing does not work with more than one GPU.  Continuing with one unspecified GPU.")
        args.gpus = 1
        args.specify_gpus = []
    main(args)
    