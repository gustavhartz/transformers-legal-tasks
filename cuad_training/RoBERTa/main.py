
# Run training of the model using pytorch lightning
from utils import get_pred_from_batch_outputs
from lightning import PLQAModel
from models import QAModel
import pytorch_lightning as pl
import torch
from data import CUADDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from transformers import AutoTokenizer


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
            tokenizer = pl_module.tokenizer

            collected_results = get_pred_from_batch_outputs(
                batch, outputs['pred'][0], outputs['pred'][1], tokenizer)
            flattend_results = [
                list(item) for sublist in collected_results for item in sublist]

            columns = ['id', 'top_k_id', 'is_impossible', 'prediction', 'answer',
                       'confidence', 'start_token_pos', 'end_token_pos']

            wandb_logger.log_text(
                key='train_pred_sample', columns=columns, data=flattend_results)

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # Output on the first batch and on every n steps - only on main GPU. Gather all not working as exptected
        if ((batch_idx == 0) or (batch_idx % hparams['log_text_every_n_batch_valid'] == 0)) and trainer.is_global_zero:
            wandb_logger = pl_module.logger

            collected_results = outputs['top_k_preds']
            flattend_results = [
                list(item) for sublist in collected_results for item in sublist]

            columns = ['id', 'top_k_id', 'is_impossible', 'prediction', 'answer',
                       'confidence', 'start_token_pos', 'end_token_pos']

            wandb_logger.log_text(
                key='valid_pred_sample', columns=columns, data=flattend_results)


def main():
    global hparams
    hparams = {
        'lr': 1e-5,
        'batch_size': 8,
        'num_workers': 5,
        'num_labels': 2,
        'hidden_size': 768,
        'num_train_epochs': 6,
        'model': 'Rakib/roberta-base-on-cuad',
        'log_text_every_n_batch': 30,
        'log_text_every_n_batch_valid': 10
    }
    train_encodings = torch.load("./data/train_encodings")
    # Use test set as validation set for now
    val_encodings = torch.load("./data/test_encodings")
    train_dataset = CUADDataset(train_encodings)
    val_dataset = CUADDataset(val_encodings)

    wandb_logger = WandbLogger(project="roberta-cuad-huggingface", entity="gustavhartz")

    print("Batch size", hparams.get("batch_size"))
    train_loader = DataLoader(
        train_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.get("batch_size"), shuffle=True, num_workers=hparams.get('num_workers'))

    del train_encodings
    del val_encodings
    model = QAModel(hparams)
    tokenizer = AutoTokenizer.from_pretrained(
        hparams['model'])
    litModel = PLQAModel(model, hparams, tokenizer)
    trainer = pl.Trainer(gpus=1, max_epochs=hparams['num_train_epochs'],
                         logger=wandb_logger, strategy='ddp', callbacks=[LogTextSamplesCallback()])
    trainer.fit(litModel, train_loader, val_loader)
    torch.save(litModel.model, "model.model")


if __name__ == "__main__":
    main()
