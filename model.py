import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)  # match saved model
        self.fc = nn.Linear(128, vocab_size)        # match saved model

    def forward(self, x):
        x = self.embed(x)
        return self.fc(x)

