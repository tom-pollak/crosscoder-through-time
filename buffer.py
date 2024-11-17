# %%
from typing import Iterator
from jaxtyping import Float
import torch as t
from datasets import IterableDataset, load_dataset


class MultiFeatureBuffer:
    def __init__(
        self,
        repo_id,
        columns: list[str] | None = None,  # all columns if None
        shuffle: bool = True,
        seed: int = 42,
        **dataset_kwargs,
    ):
        """
        Buffer for a dataset with multiple features from the same input sequence.

        All features are flattened into a single batch dimension.

        All features must have the same model dimension (and sequence length).
        """
        self.ds = load_dataset(repo_id, split="train", streaming=True, **dataset_kwargs)
        assert isinstance(self.ds, IterableDataset)
        self.ds = self.ds.with_format(type="torch")
        if columns is not None:
            self.ds = self.ds.select_columns(columns)
        if shuffle:
            self.ds = self.ds.shuffle(seed=seed)

        feats = self.ds.features
        assert feats is not None
        self.seq_len, self.d_model = feats[list(feats.keys())[0]].shape

    @staticmethod
    def flatten_activations(
        batch: dict[str, Float[t.Tensor, "n_rows seq_len d_model"]],
    ) -> Float[t.Tensor, "(n_rows * seq_len) n_models d_model"]:
        """
        Takes a dataset with columns of different features from the same input sequence
        {model: [n_rows, seq_len, d_model]}

        And transforms into a single batch with shape
        [(n_rows * seq_len), n_models, d_model]

        Where the new batch size is n_rows * seq_len
        """
        # n_models (n_rows * seq_len) d_model
        acts = t.stack([t.flatten(batch[model], 0, 1) for model in batch])
        # (n_rows * seq_len) n_models d_model
        return acts.permute(1, 0, 2)

    def iter(
        self, batch_size: int, drop_last_batch: bool = True
    ) -> Iterator[Float[t.Tensor, "model batch_size d_model"]]:
        assert batch_size % self.seq_len == 0, "batch_size must be divisible by seq_len"
        assert batch_size % self.d_model == 0, "batch_size must be divisible by d_model"
        row_batch_size = batch_size // self.seq_len
        return map(
            self.flatten_activations,
            self.ds.iter(batch_size=row_batch_size, drop_last_batch=drop_last_batch),  # type: ignore
        )


if __name__ == "__main__":
    repo_id = "tommyp111/pythia-70m-layer-4-pile-resid-post-activations-through-time"
    buffer = MultiFeatureBuffer(repo_id)
    it = buffer.iter(batch_size=1024, drop_last_batch=True)
    batch = next(it)
    print(batch.shape)
    batch = next(it)
    print(batch.shape)
