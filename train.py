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

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def estimate_loss(logits, gt):
    B, T, C = logits.shape

    logits = logits.view(B*T, C)
    gt = gt.view(B*T)
    loss = F.cross_entropy(logits, gt)
    return loss

@torch.no_grad()
def validate(epoch):
    logger.info("Validating---")
    nano_gpt.eval()
    val_losses = []
    for v_idx, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        y = y.to(device)
        preds = nano_gpt(x)
        loss = estimate_loss(preds, y)
        val_losses.append(loss.item())
    val_losses = torch.tensor(val_losses)
    val_losses = val_losses.mean()
    nano_gpt.train()

    logger.info(f"val loss @ epoch- {epoch}, loss - {val_losses.item()}")


def train_epoch(epoch):
    for t_idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)

        preds = nano_gpt(x)
        loss = estimate_loss(preds, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if t_idx % log_freq == 0:
            logger.info(f"epoch - {epoch}, iter - {t_idx} / {train_dataloader.__len__()}, train_loss - {loss.item()}")


def run_training():
    for epoch in tqdm(range(max_epochs)):
        train_epoch(epoch)
        if epoch % val_freq == 0:
            validate(epoch)
            logger.info("sample generation after epoch -- {epoch}---------------------------------------------")
            context = val_dataset.encode("Dumbledore turned and walked")
            context = torch.tensor(context, dtype=torch.long).unsqueeze(-1)
            model_gen = nano_gpt.generate(context.to(device), max_new_tokens=100)[0].tolist()
            print(train_dataset.decode(model_gen))
            logger.info("--------------------------------------------------------------------------------------")

            torch.save(nano_gpt.state_dict(), f"checkpoints/{epoch}-ckpt.pth")


if __name__ == "__main__":
    # todo: use lightning to create proper train and val process

    # hyperparameters
    val_freq = 2
    log_freq = 10000
    batch_size = 2  # how many independent sequences will we process in parallel?
    block_size = 32  # what is the maximum context length for predictions?
    max_epochs = 10
    max_iters = 5000
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 64
    n_head = 8
    n_layer = 8

    torch.manual_seed(1337)

    train_dataset = HarryPotterDataSet("Harry_Potter_all_books_preprocessed.txt", "train",
                                       block_size, tokenizer="char")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)

    val_dataset = HarryPotterDataSet("Harry_Potter_all_books_preprocessed.txt", "val",
                                     block_size, tokenizer="char")
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)

    # reading model configs
    with open("configs/model_config.yaml", "r") as mcf:
        model_configs = yaml.safe_load(mcf)

    model_configs["vocab_size"] = train_dataset.vocab_size
    # creating model instance
    nano_gpt = NanoGPT(model_configs)
    nano_gpt.to(device)

    #context = train_dataset.encode("Dumbledore turned and walked")
    #context = torch.tensor(context, dtype=torch.long, device="cuda").unsqueeze(-1)
    # model_gen = nano_gpt.generate_greedy(context, max_new_tokens=50).tolist()
    #model_gen = nano_gpt.generate_beam_search(context, max_new_tokens=50, beam_size=3).tolist()
    #print(train_dataset.decode(model_gen))

    optimizer = torch.optim.AdamW(nano_gpt.parameters(), learning_rate)

    run_training()