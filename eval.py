"""
provides utility to use trained model to generate text
"""
from tqdm import tqdm
from loguru import logger
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader

from dataset import HarryPotterDataSet
from model import NanoGPT


def calculate_perplexity(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # Flatten the predicted logits and ground truth labels
    pred = pred.view(-1, pred.size(-1))  # (batch_size * seq_len, vocab_size)
    gt = gt.view(-1)  # (batch_size * seq_len)

    # Compute cross-entropy loss (negative log likelihood)
    loss = F.cross_entropy(pred, gt, reduction='mean')

    # Perplexity is the exponent of the loss
    perplexity = torch.exp(loss)

    return perplexity


def eval():
    perplexity = []
    for x, y in tqdm(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        preds = nano_gpt(x)
        pp = calculate_perplexity(preds, y)
        perplexity.append(pp.item())

    perplexity = torch.tensor(perplexity).mean()
    return perplexity.item()

if __name__ == "__main__":
    log_freq = 1000
    batch_size = 1 # how many independent sequences will we process in parallel?
    block_size = 32  # what is the maximum context length for predictions?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 64
    n_head = 8
    n_layer = 8

    torch.manual_seed(1337)
    # create test dataset
    test_dataset = HarryPotterDataSet("Harry_Potter_all_books_preprocessed.txt", "test",
                                       block_size, tokenizer="tiktoken")
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)

    # instantiate model
    nano_gpt = NanoGPT(n_layer, n_embd, n_head, block_size, test_dataset.vocab_size)
    # load ckpt
    nano_gpt.load_state_dict(torch.load("checkpoints/2-ckpt.pth", weights_only=True))
    nano_gpt.eval()
    nano_gpt.to(device)

    ppx = eval()
    logger.info("Perplexity on test dataset - {ppx}")


