import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
import tiktoken

class PileDataset(IterableDataset):
    def __init__(self, split="train", context_length=128):
        self.dataset = load_dataset("monology/pile-uncopyrighted", split=split, streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.context_length = context_length
        self.current_tokens = []

    def __iter__(self):
        for item in self.dataset:
            text = item['text'] + "<|endoftext|>"
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            self.current_tokens.extend(tokens)
            
            while len(self.current_tokens) >= self.context_length:
                yield torch.tensor(self.current_tokens[:self.context_length], dtype=torch.long)
                self.current_tokens = self.current_tokens[self.context_length:]