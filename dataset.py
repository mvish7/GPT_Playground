"""
provides utilities for dataset prep and dataloading
"""
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class HarryPotterDataSet:
    def __init__(self, path, split, block_size, tokenizer):
        """
        implements a dataset for harry potter books
        :param path: path to the .txt file
        :param split: data split among train, test and val
        :param block_size: num token to be considered at once
        :param tokenizer: type of tokenizer to use
        """
        super().__init__()

        self.path = path
        self.data_split = split
        self.block_size = block_size
        # placeholders
        self.text = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self._open_txt()

        if tokenizer == "tiktoken":
            encoding = tiktoken.get_encoding("o200k_base")
            tokens = encoding.encode(self.text)
            self.vocab_size = encoding.n_vocab
            self.encode = encoding.encode
            self.decode = encoding.decode
            self.decode_single_byte = encoding.decode_single_token_bytes
        elif tokenizer == "char":
            # simplest character level tokenization
            chars = sorted(list(set(self.text)))
            self.vocab_size = len(chars)

            # creating a mapping of characters to integers
            self.stoi = {ch: idx for idx, ch in enumerate(chars)}
            self.itos = {idx: ch for idx, ch in enumerate(chars)}

            # encode -- converts a char to int, decode -- vice versa
            self.encode = lambda s: [self.stoi[c] for c in s]
            self.decode = lambda l: "".join([self.itos[i] for i in l])

        # converting the text to numbers using the encode function
        self.data = self.encode(self.text)
        self.data = torch.tensor(self.data, dtype=torch.long)
        # create dataset splits
        self._split_data()

    def _open_txt(self):
        with open(self.path, "r") as d_ip:
            self.text = d_ip.read()

    def _split_data(self):
        # splitting the data in test train sets
        train_len = int(0.9 * len(self.data))
        self.train_data = self.data[:train_len]
        remaining = self.data[train_len:]
        val_len = int(0.2 * len(self.data))
        self.val_data = remaining[:val_len]
        self.test_data = remaining[val_len:]

    def __getitem__(self, idx):
        data = self._identify_data()
        # ix = torch.randint(len(data)-self.block_size, (4,))
        x = data[idx:idx+self.block_size]
        y = data[idx+1:idx+self.block_size+1]
        return x, y

    def __len__(self):
        data = self._identify_data()
        return len(data) - self.block_size

    def _identify_data(self):
        data = None
        if self.data_split == "train":
            data = self.train_data
        elif self.data_split == "val":
            data = self.val_data
        elif self.data_split == "test":
            data = self.test_data
        else:
            raise TypeError("dataset split should be in train, val or test")
        return data


if __name__ == "__main__":
    text_dataset = HarryPotterDataSet("Harry_Potter_all_books_preprocessed.txt", "train", 12,
                                      tokenizer="char")
    text_dataloader = DataLoader(dataset=text_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    x, y = next(iter(text_dataloader))
    print(x)

