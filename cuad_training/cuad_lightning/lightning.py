import json
import os
import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from helpers import make_dataset_path
from utils import (
    compute_predictions_logits,
    squad_evaluate,
)
from evaluate import get_results
from utils_valid import (get_pred_from_batch_outputs,
                         compute_top_1_scores_from_preds)
import logging
import random
from tqdm import tqdm
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


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
        self.log("train_loss", loss, sync_dist=True)
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
        self.log("valid_loss", loss, sync_dist=True)

        top_k_preds = get_pred_from_batch_outputs(
            self.args, batch, outputs[1], outputs[2], self.tokenizer)

        rs = compute_top_1_scores_from_preds(top_k_preds)

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l, 'metrics': rs}

    def validation_epoch_end(self, outputs):
        ct_batch, ct_total = 0, 0
        em_sum, f1_sum, _loss_sum = 0, 0, 0
        tp, fp, fn, tn = 0, 0, 0, 0
        for pred in outputs:
            _loss_sum += pred['loss'].item()
            em_sum += pred['metrics']['em']
            f1_sum += pred['metrics']['f1']
            ct_batch += 1
            ct_total += pred['metrics']['batch_len']

            tp += pred['metrics']['tp']
            fp += pred['metrics']['fp']
            fn += pred['metrics']['fn']
            tn += pred['metrics']['tn']

        self.log(
            "epoch_valid_loss",
            _loss_sum / ct_batch,
            sync_dist=True
        )

        performance_stats = {
            'tp': tp if tp else -1,
            'fp': fp if fp else -1,
            'fn': fn if fn else -1,
            'tn': tn if tn else -1,
            'recall': 100*tp/(tp+fn) if (tp+fn) > 0 else 0,
            'precision': 100*tp/(tp+fp) if (tp+fp) > 0 else 0,
            'observations': ct_total if ct_total else -1,
            'em': 100*em_sum/ct_total if (100*em_sum/ct_total) > 0 else 0,
            'f1_batch': 100*f1_sum/ct_batch if (100*f1_sum/ct_batch) > 0 else 0,
            'f1_total': 100*f1_sum/ct_total if (100*f1_sum/ct_total) > 0 else 0,
        }
        for k, v in performance_stats.items():
            self.log("performance_stat_"+k, float(v)
                     if isinstance(v, int) else v, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
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

        self.log(
            "test_loss",
            loss,
            sync_dist=True
        )

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l}

    def test_epoch_end(self, outputs):

        DATASET_PATH = make_dataset_path(self.args, True)
        examples = torch.load(DATASET_PATH+"_examples")
        features = torch.load(DATASET_PATH+"_features")
        all_results = []
        for pred in outputs:
            for i, feature_index in enumerate(pred['feature_indices']):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                start_logits = pred['start_logits'][i]
                end_logits = pred['end_logits'][i]
                all_results.append(
                    SquadResult(unique_id, start_logits, end_logits))

        # Save predictions
        output_prediction_file = DATASET_PATH + f"_predictions.json"
        output_nbest_file = DATASET_PATH + f"_nbest_predictions.json"
        output_null_log_odds_file = DATASET_PATH + f"_null_odds.json"
        with open(self.args.predict_file, "r") as f:
            json_test_dict = json.load(f)

        logging.info("Calculating predictions")
        predictions = compute_predictions_logits(
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
            False,  # self.args.verbose_logging,
            True,  # self.args.version_2_with_negative
            self.args.null_score_diff_threshold,
            self.tokenizer,
        )
        logging.info("Evaluating predictions")
        # Handle results
        results = squad_evaluate(examples, predictions)
        logging.info("Getting results")
        res = get_results(self.args, output_nbest_file,
                          gt_dict=json_test_dict, include_model_info=False)

        if self.args.verbose:
            logging.info("***** Eval results *****")
            print(results)
            print(res)

        for k, v in results.items():
            if not v:
                logging.warn(
                    f"In logging performance_stats_test: {k} got value {v}")
            self.log("performance_stats_test"+k, float(v)
                     if isinstance(v, int) else v)

        for k, v in res.items():
            if not v:
                logging.warn(
                    f"In logging performance_AUPR_test: {k} got value {v}")
            self.log("performance_AUPR_test"+k, float(v)
                     if isinstance(v, int) else v)

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
