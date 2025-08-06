def get_config():
    return {
        "vocab_size": 50257,
        "embedding_dim": 128,
        "seq_len": 128,
        "num_layers": 2,
        "batch_size": 8,
        "epochs": 10,
        "learning_rate": 3e-4,
    }
