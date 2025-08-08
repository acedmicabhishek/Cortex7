import torch
import torch.optim as optim
from models.cortex_model import CortexModel
from data.dataset_loader import get_dummy_dataset

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, vocab_size = get_dummy_dataset(config["seq_len"], config["batch_size"])
    config["vocab_size"] = vocab_size
    model = CortexModel(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["epochs"]):
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
