"""
provides utilities to train nano-GPT
"""
import os
from tqdm import tqdm
from loguru import logger
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import HarryPotterDataSet
from model.nano_gpt import NanoGPT
from losses.token_prediction_losses import single_token_pred_loss, multi_token_pred_loss

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Trainer:

    def __init__(self, train_config_path, model_config_path):
        with open(train_config_path, "r") as f:
            self.train_config = yaml.safe_load(f)

        with open(model_config_path, "r") as mf:
            self.model_config = yaml.safe_load(mf)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(self.train_config['training']['seed'])

        self._setup_data()
        self._setup_model()
        self._setup_loss_fun()
        self._setup_optimizer()

    def _setup_data(self):
        train_config = self.train_config['training']
        data_config = self.train_config['data']

        self.train_dataset = HarryPotterDataSet(
            data_config['train_file'],
            "train",
            train_config['block_size'],
            tokenizer=data_config['tokenizer'])

        self.train_dataloader = DataLoader(
            self.train_dataset,
            train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'])

        self.val_dataset = HarryPotterDataSet(
            data_config['val_file'],
            "val",
            train_config['block_size'],
            tokenizer=data_config['tokenizer'])

        self.val_dataloader = DataLoader(
            self.val_dataset,
            train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'])

    def _setup_loss_fun(self):
        if self.model_config["multi_token_pred"]["do_mtp"]:
            self.loss_fun = multi_token_pred_loss
        else:
            self.loss_fun = single_token_pred_loss

    def _setup_model(self):

        self.model_config["vocab_size"] = self.train_dataset.vocab_size
        self.model = NanoGPT(self.model_config)
        self.model.to(self.device)

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), self.train_config['training']['learning_rate'])

    @torch.no_grad()
    def validate(self, epoch):
        logger.info("Validating---")
        self.model.eval()
        val_losses = []

        for v_idx, (x, y) in enumerate(self.val_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.model(x)
            loss = self.loss_fun(preds, y)
            val_losses.append(loss.item())

        val_losses = torch.tensor(val_losses).mean()
        self.model.train()
        logger.info(f"val loss @ epoch- {epoch}, loss - {val_losses.item()}")
        return val_losses

    def train_epoch(self, epoch):
        for t_idx, (x, y) in enumerate(self.train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)

            preds = self.model(x)
            loss = self.loss_fun(preds, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if t_idx % self.train_config['training']['log_freq'] == 0:
                logger.info(
                    f"epoch - {epoch}, iter - {t_idx} / {len(self.train_dataloader)}, "
                    f"train_loss - {loss.item()}")

    def generate_sample(self,
                        context_text="Dumbledore turned and walked",
                        max_tokens=100):
        context = self.val_dataset.encode(context_text)
        context = torch.tensor(context, dtype=torch.long).unsqueeze(-1)
        model_gen = self.model.generate(context.to(self.device),
                                        max_new_tokens=max_tokens)[0].tolist()
        return self.train_dataset.decode(model_gen)

    def run_training(self):
        for epoch in tqdm(range(self.train_config['training']['max_epochs'])):
            self.train_epoch(epoch)

            if epoch % self.train_config['training']['val_freq'] == 0:
                self.validate(epoch)
                logger.info(
                    f"sample generation after epoch -- {epoch}---------------------------------------------"
                )
                print(self.generate_sample())
                logger.info(
                    "--------------------------------------------------------------------------------------"
                )

                torch.save(self.model.state_dict(),
                           f"checkpoints/{epoch}-ckpt.pth")


if __name__ == "__main__":
    trainer = Trainer(train_config_path="configs/training_config.yaml", model_config_path="configs/model_config.yaml")
    trainer.run_training()
