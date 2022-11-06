from collections import defaultdict
import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from helpers import make_dataset_path, make_prediction_path
import json
import torch.distributed as dist
import time
from utils_v2 import compute_predictions_logits_multi
from utils import squad_evaluate, squad_evaluate_nbest
from evaluate import get_results
import logging
import pandas as pd
import numpy as np
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

examples = None
features = None
json_test_dict = None


class PLQAModel(pl.LightningModule):
    def __init__(self, model, args, hparams, tokenizer):
        super().__init__()
        self.hparams.update(hparams)
        self.args = args
        self.model = model
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        # bool to check if the validation returns a loss - Logging
        self.val_has_loss = True

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        batch = x
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        return outputs

    # TRAINING
    def training_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        loss = outputs[0].get("total_loss")
        # self.log({"train_"+k: v for k, v in outputs[0].items()})
        self.log("train_", outputs[0])
        return {'loss': loss, 'pred': [outputs[1], outputs[2]], "loss_dict": outputs[0]}

    def training_epoch_end(self, outputs):
        # Convert dict of losses to array with "key" rows and "len(all_outputs)" columns
        loss_d = defaultdict(list)
        for x in outputs:
            assert "loss_dict" in x, "Using model that does not return a loss dict"
            for k, v in x["loss_dict"].items():
                loss_d[k].append(v.item())

        self.log("epoch_train", {k: np.mean(v) for k, v in loss_d.items()})

    # VALIDATION

    def validation_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        feature_indices = batch[-1]

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        s_l = outputs[1]
        e_l = outputs[2]

        loss = outputs[0].get("total_loss")
        self.log("valid_", outputs[0])

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l, "loss_dict": outputs[0]}

    def validation_epoch_end(self, outputs):
        all_outputs = []
        start_time = time.time()
        for oup in outputs:
            oup_collected = self.all_gather(oup)
            # Un-nest potential dicts
            new_dict_collected = {}
            for k, v in oup_collected.items():
                if type(v) == dict:
                    new_dict_collected = {**new_dict_collected, **v}
                else:
                    new_dict_collected[k] = v
            # Dope way of converting it into a list of dicts
            all_outputs.extend([dict(zip(new_dict_collected.keys(), x))
                               for x in list(zip(*new_dict_collected.values()))])
        loss_keys = [k for k in all_outputs[0].keys() if "loss" in k]
        loss_dict = defaultdict(list)
        # N x LossTypes
        for ele in all_outputs:
            for k in loss_keys:
                loss_dict[k].append(ele[k])
        for k, v in loss_dict.items():
            # Epoch_valid is added to name of key when logging
            mean = torch.mean(torch.stack(v))
            self.log("epoch_valid_"+k, mean, rank_zero_only=True)
            print(k, mean)

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:

            logging.info(
                f"Collected predictions from all the nodes. Processing time {time.time() - start_time}")

            # lazy load in data
            global examples
            global features
            global json_test_dict
            DATASET_PATH = make_dataset_path(self.args, True)
            PRED_FILES_PATH = DATASET_PATH + "_model-name_" + self.args.model_name
            if not examples or not features:
                start_time = time.time()
                logging.warning(
                    f"Loading examples, features, and predictions on main process")
                features = torch.load(DATASET_PATH+"_features")
                examples = torch.load(DATASET_PATH+"_examples")
                with open(self.args.predict_file, "r") as f:
                    json_test_dict = json.load(f)

            # Create the squad_result object
            all_results = []
            for pred in all_outputs:
                for i, feature_index in enumerate(pred['feature_indices']):
                    eval_feature = features[feature_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    start_logits = pred['start_logits'][i]
                    end_logits = pred['end_logits'][i]
                    all_results.append(
                        SquadResult(unique_id, start_logits.cpu(), end_logits.cpu()))
            # Calculate scores
            DATASET_PATH = make_dataset_path(self.args, True)
            PRED_FILES_PATH = make_prediction_path(self.args, True) + "_model-name_" + \
                self.args.model_name + \
                f"_global-step_{self.trainer.global_step}_epoch{self.trainer.current_epoch}"
            output_prediction_file = PRED_FILES_PATH + f"_predictions.json"
            output_nbest_file = PRED_FILES_PATH + f"_nbest_predictions.json"
            output_null_log_odds_file = PRED_FILES_PATH + f"_null_odds.json"

            logging.info("Calculating predictions")
            predictions = compute_predictions_logits_multi(
                json_test_dict,
                examples,
                features,
                all_results,
                self.args.n_best_size,
                self.args.max_answer_length,
                self.args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                self.args.verbose,
                True,  # self.args.version_2_with_negative
                self.args.null_score_diff_threshold,
                self.tokenizer,
                threads=self.args.test_examples_workers,
                chunk_size=self.args.test_examples_chunk_size,
                pred_logic=self.args.prediction_logic,
            )
            logging.info("Evaluating predictions")
            ##################
            # Handle results #
            ##################

            # Keept to ensure correct logic of the evaluation in n_best
            results = squad_evaluate(examples, predictions)

            results_1, exact_1, f1_1 = squad_evaluate_nbest(
                examples, output_nbest_file, n_best_size=1, return_dict=True).values()

            results_n_best, exact_n_best, f1_n_best = squad_evaluate_nbest(
                examples, output_nbest_file, n_best_size=self.args.n_best_size_squad_evaluate, return_dict=True).values()

            res = get_results(self.args, output_nbest_file,
                              gt_dict=json_test_dict, include_model_info=False)
            logging.info("Results obtained")

            try:
                assert results_1 == results
            except AssertionError:
                logging.error(
                    f"Results do not match for n_best_size=1 and original eval framework")
                logging.error(f"Original results: {results}")
                logging.error(f"New results: {results_1}")

            #########################################################################################
            # Question category level results - Adopted from performance across categories notebook #
            #########################################################################################
            data = []
            for k, v in exact_1.items():
                # question name, question type, value, category
                data.append([k, k.split("__")[1], v, "em nbest=1"])

            for k, v in f1_1.items():
                # question name, question type, value, category
                data.append([k, k.split("__")[1], v, "f1 nbest=1"])

            # add the nbest=3 to df
            for k, v in exact_n_best.items():
                # question name, question type, value, category
                data.append([k, k.split("__")[1], v, "em nbest=3"])

            for k, v in f1_n_best.items():
                # question name, question type, value, category
                data.append([k, k.split("__")[1], v, "f1 nbest=3"])

            df = pd.DataFrame(
                data, columns=["question_name", "question_type", "value", "category"])
            df = df.groupby(["question_type", "category"]).mean().reset_index()

            logging.info(f"Finished evaluating predictions rank zero")

        # Force sync between processes related to logging
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11242
        logging.info(
            f"Reached pl sync barrier on rank: {self.trainer.global_rank}")
        dist.barrier()
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            for k, v in results.items():
                self.log(f"top_1"+k, float(v)
                         if isinstance(v, int) else v, rank_zero_only=True)

            for k, v in results_n_best.items():
                self.log(f"nbest_"+k, float(v)
                         if isinstance(v, int) else v, rank_zero_only=True)

            for k, v in res.items():
                self.log(k, float(v)
                         if isinstance(v, int) else v, rank_zero_only=True)
            for idx, row in df.iterrows():
                self.log("spec_"+row.question_type+row.category, float(row.value)
                         if isinstance(v, int) else 0, rank_zero_only=True)
            logging.info(f"Finished logging rank zero dist barrier")
        logging.info(
            f"Passed pl sync barrier on rank: {self.trainer.global_rank}")
        dist.barrier()
        logging.info(
            f"Done with validation epoch code on rank: {self.trainer.global_rank}")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * \
            float(self.trainer.max_epochs)
        self.total_steps = int(
            (self.hparams.train_set_size // tb_size) * ab_size)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler,
                     'name': 'learning_rate',
                     'interval': 'step',
                     'frequency': 1}
        return dict(optimizer=optimizer, lr_scheduler=scheduler)
