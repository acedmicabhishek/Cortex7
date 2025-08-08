import torch.nn as nn

class CortexBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["embedding_dim"], config["embedding_dim"])
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dense(x))
