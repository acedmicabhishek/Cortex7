```
Cortex7/
├── config/
│   └── config.yaml          # Hyperparams, model settings
├── data/
│   ├── dataset_loader.py    # Tokenizer + Dataset pipeline
│   └── preprocessor.py      # Text cleaning, sequence padding
├── models/
│   ├── cortex_block.py      # custom architecture unit
│   ├── cortex_model.py      # Model build logic
│   └── layers.py            # Custom TF layers (e.g., Fourier, Memory)
├── train/
│   ├── trainer.py           # Training loop, optimizer, checkpointing
│   └── loss.py              # Custom loss functions (e.g., label smoothing)
├── eval/
│   ├── evaluator.py         # BLEU, perplexity, top-k accuracy, etc.
│   └── sample_gen.py        # Text generation samples
├── utils/
│   ├── logger.py            # Custom training logger
│   └── schedule.py          # Custom LR schedules (warmup, decay)
├── checkpoints/
│   └── ...                  # Saved weights, best models
├── run.py                   # Entry point for training
└── README.md
```