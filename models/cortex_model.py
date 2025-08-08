import torch.nn as nn
from models.cortex_block import CortexBlock

class CortexModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.blocks = nn.ModuleList([CortexBlock(config) for _ in range(config["num_layers"])])
        self.final_ln = nn.LayerNorm(config["embedding_dim"])
        self.head = nn.Linear(config["embedding_dim"], config["vocab_size"])

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.head(x)
