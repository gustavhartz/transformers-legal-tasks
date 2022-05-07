import argparse
import os
import random
import string
import numpy as np
import torch


def make_dataset_name_base(args):
    """Make the base path for the dataset

    Args:
        args (_type_): Argparse
    """
    dataset_base = f"dataset-name_{args.dataset_name}_"+f"model-type_{args.model_type}_" + \
        f"only-first-ans_{str(args.only_first_answer_in_features)}_" + \
        f"doc-stride_{str(args.doc_stride)}_dataset-type"
    return dataset_base


def make_dataset_name(args, evaluate: bool) -> str:
    dataset_name_base = make_dataset_name_base(args)
    return dataset_name_base + "_eval_" + "predict-file-version_"+args.predict_file_version if evaluate else dataset_name_base + "_train_" + "train-file-version_" + args.train_file_version


def make_dataset_path(args, evaluate: bool) -> str:
    DATASET_NAME = make_dataset_name(args, evaluate)
    dataset_path = os.path.join(
        args.out_dir, DATASET_NAME)
    return dataset_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function that generates random string of length nª


def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))


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
