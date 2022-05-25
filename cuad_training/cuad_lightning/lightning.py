import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from helpers import make_dataset_path, make_prediction_path
import json
import torch.distributed as dist
import time
from utils_v2 import compute_predictions_logits_multi
from utils import squad_evaluate
from evaluate import get_results
import wandb
import logging
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
        loss = outputs[0]
        self.log("train_loss", loss)
        return {'loss': loss, 'pred': [outputs[1], outputs[2]]}

    def training_epoch_end(self, outputs):
        ct, _sum = 0, 0
        for pred in outputs:
            _sum += pred['loss'].item()
            ct += 1
        self.log(
            "epoch_train_loss",
            _sum / ct,
            sync_dist=True
        )

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
        loss = outputs[0]
        s_l = outputs[1]
        e_l = outputs[2]
        self.log("valid_loss", loss)

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l}

    def validation_epoch_end(self, outputs):
        all_outputs = []
        start_time = time.time()
        loss = 0
        for oup in outputs:
            oup_collected = self.all_gather(oup)
            if oup_collected['loss'].shape != oup['loss'].shape:
                all_outputs.extend([{'loss': x[0], 'start_logits': x[1], 'feature_indices': x[2],
                                   'end_logits':x[3]} for x in zip(*oup_collected.values())])
            else:
                all_outputs.extend([oup_collected])

        # To avoid errors on checkpoint callback

        loss = sum([x['loss'].item()
                    for x in all_outputs]) / len(all_outputs)
        self.log("epoch_valid_loss", loss, rank_zero_only=True)

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:

            logging.info(
                f"Collected predictions from all the nodes. Processing time {time.time() - start_time}")

            # PL issue 11242
            # self.log("epoch_valid_collected_output_time", time.time() - start_time)
            wandb.log({"epoch_valid_collected_output_time":
                       time.time() - start_time})

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

                # PL issue 11242
                wandb.log({"epoch_valid_load_data_time":
                           time.time() - start_time})
                # self.log("epoch_valid_load_data_time", time.time() - start_time)

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
            )
            logging.info("Evaluating predictions")
            # Handle results
            results = squad_evaluate(examples, predictions)
            logging.info("Getting results")
            res = get_results(self.args, output_nbest_file,
                              gt_dict=json_test_dict, include_model_info=False)

            # Report metrics
            if self.args.verbose:
                logging.info("***** Eval results *****")
                print(results)
                print(res)

            post_fix = "valid"
            if self.args.test_model:
                post_fix = "test"

            # PL issue 11242
            # for k, v in results.items():
            #     wandb.log({f"performance_stats_{post_fix}_"+k: float(v)
            #               if isinstance(v, int) else v})

            # for k, v in res.items():
            #     wandb.log({f"performance_AUPR_{post_fix}_"+k: float(v)
            #                if isinstance(v, int) else v})

            logging.info(f"Finished evaluating predictions rank zero")

        # Force sync between processes related to logging
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/11242
        logging.info(
            f"Reached pl sync barrier on rank: {self.trainer.global_rank}")
        dist.barrier()
        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            for k, v in results.items():
                self.log(f"performance_stats_{post_fix}_"+k, float(v)
                         if isinstance(v, int) else v, rank_zero_only=True)

            for k, v in res.items():
                self.log(f"performance_AUPR_{post_fix}_"+k, float(v)
                         if isinstance(v, int) else v, rank_zero_only=True)
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
