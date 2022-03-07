
# Run training of the model using pytorch lightning
from lightning import PLQAModel
from models import QAModelBert
import pytorch_lightning as pl
import torch
from data import CUADDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from transformers import BertTokenizerFast


class LogTextSamplesCallback(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        # Output on the first batch and on every n steps - only on main GPU. Gather all not working as exptected
        if ((batch_idx == 0) or (batch_idx % hparams['log_text_every_n_batch'] == 0)) and trainer.is_global_zero:
            wandb_logger = pl_module.logger
            tokenizer = BertTokenizerFast.from_pretrained(
                hparams['bert_model'])
            # Make iterable and reusable where each row consitutes a set of values
            batch_values = list(zip(
                batch['input_ids'], batch['token_type_ids'], batch['start_positions'], batch['end_positions']))
            # Non zero values correspond to the seperation
            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
            questions = [tokenizer.decode(
                x[0][x[1].nonzero().squeeze()]) for x in batch_values]
            answers = [tokenizer.decode(x[0][x[2]:x[3]]) for x in batch_values]

            # Only top 1 prediction
            start_pred = torch.argmax(outputs['pred'][0], dim=1)
            end_pred = torch.argmax(outputs['pred'][1], dim=1)
            # decode
            preds_text = [tokenizer.decode(x[2][x[0]:x[1]]) if x[0] < x[1] else "[END BEFORE START]" for x in zip(
                start_pred, end_pred, batch['input_ids'])]
            columns = ['index','is_impossible','question', 'answer',
                       'prediction', 'start_pos', 'end_pos']
            data = list(
                zip(batch['id'],batch['is_impossible'], questions, answers, preds_text, start_pred, end_pred))
            wandb_logger.log_text(
                key='train_pred_sample', columns=columns, data=data)

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # Output on the first batch and on every n steps - only on main GPU. Gather all not working as exptected
        if ((batch_idx == 0) or (batch_idx % hparams['log_text_every_n_batch_valid'] == 0)) and trainer.is_global_zero:
            wandb_logger = pl_module.logger
            tokenizer = BertTokenizerFast.from_pretrained(
                hparams['bert_model'])
            # Make iterable and reusable where each row consitutes a set of values
            batch_values = list(zip(
                batch['input_ids'], batch['token_type_ids'], batch['start_positions'], batch['end_positions']))
            # Non zero values correspond to the seperation
            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer
            questions = [tokenizer.decode(
                x[0][x[1].nonzero().squeeze()]) for x in batch_values]
            answers = [tokenizer.decode(x[0][x[2]:x[3]]) for x in batch_values]

            # Only top 1 prediction
            start_pred = torch.argmax(outputs['pred'][0], dim=1)
            end_pred = torch.argmax(outputs['pred'][1], dim=1)
            # decode
            preds_text = [tokenizer.decode(x[2][x[0]:x[1]]) if x[0] < x[1] else "[END BEFORE START]" for x in zip(
                start_pred, end_pred, batch['input_ids'])]
            columns = ['index','is_impossible','question', 'answer',
                       'prediction', 'start_pos', 'end_pos']
            data = list(
                zip(batch['id'],batch['is_impossible'], questions, answers, preds_text, start_pred, end_pred))
            wandb_logger.log_text(
                key='valid_pred_sample', columns=columns, data=data)


def main():
    global hparams
    hparams = {
        'lr': 1e-5,
        'batch_size': 8,
        'num_workers': 5,
        'num_labels': 2,
        'hidden_size': 768,
        'num_train_epochs': 6,
        'bert_model': 'bert-base-uncased',
        'log_text_every_n_batch': 30,
        'log_text_every_n_batch_valid': 10
    }
    train_encodings = torch.load("./data/train_encodings")
    # Use test set as validation set for now
    val_encodings = torch.load("./data/test_encodings")
    train_dataset = CUADDataset(train_encodings)
    val_dataset = CUADDataset(val_encodings)

    wandb_logger = WandbLogger(project="bert-cuad", entity="gustavhartz")

    print("Batch size", hparams.get("batch_size"))
    train_loader = DataLoader(
        train_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))

    del train_encodings
    del val_encodings
    model = QAModelBert(hparams, hparams['bert_model'])
    tokenizer = BertTokenizerFast.from_pretrained(
        hparams['bert_model'])
    litModel = PLQAModel(model, hparams, tokenizer)
    trainer = pl.Trainer(gpus=4, max_epochs=hparams['num_train_epochs'],
                         logger=wandb_logger, strategy='ddp', callbacks=[LogTextSamplesCallback()])
    trainer.fit(litModel, train_loader, val_loader)
    torch.save(litModel.model, "model.model")


if __name__ == "__main__":
    main()
