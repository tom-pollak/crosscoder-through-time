import einops
from jaxtyping import Float
import torch as t
from datasets import Dataset, load_dataset


class MultiFeatureBuffer:
    def __init__(
        self,
        repo_id,
        columns: list[str] | None = None,  # all columns if None
    ):
        """
        Buffer for a dataset with multiple features from the same input sequence.

        All features are flattened into a single batch dimension.

        All features must have the same model dimension (and sequence length).
        """
        self.ds = load_dataset(repo_id, split="train", streaming=False)
        assert isinstance(self.ds, Dataset)
        self.ds = self.ds.with_format("torch")
        if columns is not None:
            self.ds = self.ds.select_columns(columns)

        feats = self.ds.features
        assert feats is not None
        self.seq_len, self.d_model = feats[list(feats.keys())[0]].shape

    def dl(
        self, batch_size: int
    ) -> t.utils.data.DataLoader[Float[t.Tensor, "batch n_models d_model"]]:
        def _collate_fn(batch: list[dict[str, Float[t.Tensor, "seq_len d_model"]]]):
            """
            Takes a batch of a rows from dataset with columns of different features from the same input sequence

            [{model: [seq_len, d_model]}]

            And transforms into a single batch with shape
            [n_models, seq_len, d_model]
            """
            acts = t.stack([t.stack(list(r.values())) for r in batch])
            return einops.rearrange(
                acts,
                "batch n_models seq_len d_model -> (batch seq_len) n_models d_model",
            )

        assert batch_size % self.seq_len == 0, "batch_size must be divisible by seq_len"
        assert batch_size % self.d_model == 0, "batch_size must be divisible by d_model"
        row_batch_size = batch_size // self.seq_len
        return t.utils.data.DataLoader(
            self.ds,  # type: ignore
            batch_size=row_batch_size,
            collate_fn=_collate_fn,
            drop_last=True,
            num_workers=10,
            prefetch_factor=10,
            pin_memory=False,
            shuffle=True,
        )


if __name__ == "__main__":
    repo_id = "tommyp111/pythia-70m-layer-4-pile-resid-post-activations-through-time"
    buffer = MultiFeatureBuffer(repo_id)
    dl = buffer.dl(batch_size=1024)
    it = iter(dl)
    for i in range(10):
        batch = next(iter(it))
        print(batch.shape)
