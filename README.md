# CrossCoder Through Time

> Repo aims to train a CrossCoder model across multiple checkpoints of a 70M Pythia model, and see how it evolves over training.

- Adapted from [crosscoder-model-diff-replication](https://github.com/ckkissane/crosscoder-model-diff-replication)

Two ways to train the model:

1. On the fly: Cache activations as the model is trained
2. Cached: Cache activations ahead of time, then train the SAE on these cached activations.
   - Supports multi-gpu!
   - Takes up _a lot_ of space on disk

- `train.py`: Entry point, define config and hyperparameters
- `cache_acts.py`: Cache activations ahead of time

Crosscoder lib:

- `trainer.py`: Simple training loop that can use either cached or on-the-fly activations
- `model.py`: Defines the CrossCoder model
- `buffer_on_the_fly.py`: Caches activations as the model is trained
- `buffer_cached.py`: Loads cached activations from disk
