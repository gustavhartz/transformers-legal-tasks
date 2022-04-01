from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features
import argparse
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadResult, SquadV2Processor

from evaluate_helpers_cuad import (
    squad_evaluate,
    compute_predictions_logits
)


def main(args):
    """Define the model and prepare the data"""
    # THIS SECTIONS CAN BE CUSTOMIZED to work with your model
    ###################################################################################################################

    # define model and tokenizer
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, do_lower_case=args.do_lower_case, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_path,
        from_tf=bool(".ckpt" in args.model_path),
        config=config,
        cache_dir=None)

    # check if file exists with dataset
    feature_bundle_path = f'{args.out_dir}/{args.model_name}_{args.model_version}_feature_bundle'
    if not os.path.exists(feature_bundle_path):
        print("Dataset file not found. Creating new one.")
        processor = SquadV2Processor()
        print("Creating dataset...")
        examples = processor.get_dev_examples(
            args.data_dir, filename=args.predict_file)
        print("Creating features... This is a long running process and can take multiple hours")
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )
        print("Saving dataset...")
        torch.save({"features": features, "dataset": dataset, "examples": examples},
                   feature_bundle_path)
    else:
        print("Dataset file found. Loading...")
        feature_bundle = torch.load(feature_bundle_path)
        features = feature_bundle["features"]
        dataset = feature_bundle["dataset"]
        examples = feature_bundle["examples"]

    # Make dataset
    batch_size = 32
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=batch_size)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evalueate
    print("Evaluating...")
    all_results = evaluate(model, eval_dataloader, features, args)
    # Save results
    print("Saving results...")
    torch.save({"results": all_results},
               f'{args.out_dir}/{args.model_name}_{args.model_version}_all_results')

    output_prediction_file = f'{args.out_dir}/predictions_{args.model_name}_{args.model_version}.json'

    output_nbest_file = f'{args.out_dir}/nbest_predictions_{args.model_name}_{args.model_version}.json'
    output_null_log_odds_file = f'{args.out_dir}/null_odds_{args.model_name}_{args.model_version}.json'

    with open(f'{args.data_dir}/test.json', "r") as f:
        json_test_dict = json.load(f)
    # Compute predictions
    print("Computing predictions...")
    predictions = compute_predictions_logits(
        json_test_dict,
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # Verbose logging
        True,  # Version two with negatives
        args.null_score_diff_threshold,
        tokenizer,
    )
    results = squad_evaluate(examples, predictions)
    print(results)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(model, eval_dataloader, features, args):
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)
    return all_results


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

    # Model max sequence length soft limit
    argparser.add_argument('--doc_stride', type=int,
                           default=128, help='document stride')
    # Create encodings argument - boolean
    argparser.add_argument('--max_seq_length', type=int,
                           default=512, help='Max sequence length')
    # postfix for datasets
    argparser.add_argument('--data_dir', type=str,
                           default='../data', help='Data directory')
    # Used cached data
    argparser.add_argument('--cached_data', type=bool,
                           default=True, help='Use cached data')
    # Predict file
    argparser.add_argument('--predict_file', type=str,
                           default='test.json', help='Predict file')
    # Out dir
    argparser.add_argument('--out_dir', type=str,
                           default='./eval_data', help='Out dir')
    # model version
    argparser.add_argument('--model_version', type=str,
                           default='v1', help='Model version')
    # Do lower case
    argparser.add_argument('--do_lower_case', type=bool,
                           default=True, help='Do lower case')
    # max query length
    argparser.add_argument('--max_query_length', type=int,
                           default=64, help='Max query length')
    # Device
    argparser.add_argument('--device', type=str,
                           default='cpu', help='Device for inference')
    # N best size
    argparser.add_argument('--n_best_size', type=int,
                           default=1, help='N best size')

    # Max answer length
    argparser.add_argument('--max_answer_length', type=int,
                           default=200, help='Max answer length')
    # Null score diff threshold
    argparser.add_argument('--null_score_diff_threshold', type=float,
                           default=0.0, help='Null score diff threshold')

    args = argparser.parse_args()
    main(args)
