import json
import os
import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult
from utils import (
    compute_predictions_logits,
    squad_evaluate,
)
from evaluate import get_results
from utils_valid import (get_pred_from_batch_outputs,
                         compute_top_1_scores_from_preds)
import logging
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
        }
        feature_indices = batch[3]

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        if len(outputs) == 3:
            loss = outputs[0]
            s_l = outputs[1]
            e_l = outputs[2]

            self.log(
                "valid_loss",
                loss,
                sync_dist=True
            )
        else:
            logging.critical("Validation did not return loss")
            loss = 0
            s_l = outputs[0]
            e_l = outputs[1]

        top_k_preds = get_pred_from_batch_outputs(
            self.args, batch, s_l, e_l, self.tokenizer)
        rs = compute_top_1_scores_from_preds(top_k_preds)
        self.log(
            "performance",
            rs,
        )

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l, 'metrics': rs}

    def validation_epoch_end(self, outputs):
        ct_batch, ct_total, _sum = 0, 0, 0
        em_sum, f1_sum = 0, 0
        for pred in outputs:
            _sum += pred['loss'].item() if type(pred['loss']
                                                ) != int else pred['loss']
            em_sum += pred['metrics']['em']
            f1_sum += pred['metrics']['f1']
            ct_batch += 1
            ct_total += pred['metrics']['batch_len']
        self.log(
            "epoch_val_loss",
            _sum / ct_batch,
            sync_dist=True
        )
        self.log(
            "epoch_val_f1",
            f1_sum / ct_batch,
            sync_dist=True
        )
        self.log(
            "epoch_val_em",
            em_sum / ct_total,
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }
        feature_indices = batch[3]

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
            del inputs["token_type_ids"]

        outputs = self.model(**inputs)
        if len(outputs) == 3:
            loss = outputs[0]
            s_l = outputs[1]
            e_l = outputs[2]

            self.log(
                "test_loss",
                loss,
                sync_dist=True
            )
        else:
            loss = 0
            s_l = outputs[0]
            e_l = outputs[1]

        return {'loss': loss, 'start_logits': s_l, 'feature_indices': feature_indices, 'end_logits': e_l}

    def test_epoch_end(self, outputs):

        DATASET_PATH = os.path.join(self.args.out_dir, self.args.dataset_name+"_" +
                                    self.args.model_type+"_"+self.args.model_name+"_eval")
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

        # Generate random string for filename
        import random
        n = random.randint(0, 12034234)

        output_prediction_file = os.path.join(
            self.args.out_dir, self.args.model_name+f"_{self.args.model_version}"+f"_epoch_{self.current_epoch}_rand{n}"+"_eval_predictions.json")
        output_nbest_file = os.path.join(
            self.args.out_dir, self.args.model_name+f"_{self.args.model_version}"+f"_epoch_{self.current_epoch}_rand{n}"+"_eval_nbest_predictions.json")
        output_null_log_odds_file = os.path.join(
            self.args.out_dir, self.args.model_name+f"_{self.args.model_version}"+f"_epoch_{self.current_epoch}_rand{n}"+"_eval_null_odds.json")
        with open(self.args.predict_file, "r") as f:
            json_test_dict = json.load(f)

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

        results = squad_evaluate(examples, predictions)
        print(results)
        res = get_results(self.args, output_nbest_file, gt_dict=json_test_dict)
        print(res)

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
