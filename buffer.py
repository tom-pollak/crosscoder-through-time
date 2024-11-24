import warnings
from dataclasses import dataclass
from pathlib import Path
import einops
from jaxtyping import Float
import torch as t
from typing import Iterator
from datasets import Dataset, concatenate_datasets


@dataclass
class MultiFeatureBufferConfig:
    activations_path: Path | str  # type: ignore
    hook_name: str
    batch_size: int
    normalization_factor: Float[t.Tensor, "n_models"] | None
    seed: int = 42

    def __post_init__(self):
        self.activations_path = Path(self.activations_path)
        assert self.activations_path.exists() and self.activations_path.is_dir()
        self.activations_path: Path


class MultiFeatureBuffer:
    def __init__(
        self,
        cfg: MultiFeatureBufferConfig,
    ):
        """
        Buffer for a dataset with multiple features from the same input sequence.

        All features are flattened into a single batch dimension.

        All features must have the same model dimension (and sequence length).
        """

        self.cfg = cfg
        self.ds = concatenate_datasets(
            [
                Dataset.load_from_disk(p).rename_column(cfg.hook_name, p.stem)
                for p in cfg.activations_path.glob("step*")
            ],
            axis=1,
        )
        self.ds.set_format(type="torch")
        self.ds = self.ds.shuffle(seed=cfg.seed)
        assert isinstance(self.ds, Dataset)

        self.batch_size = cfg.batch_size
        feats = self.ds.features
        assert feats is not None
        self.seq_len, self.d_model = feats[list(feats.keys())[0]].shape

        assert (
            self.batch_size % self.seq_len == 0
        ), "batch_size must be divisible by seq_len"
        assert (
            self.batch_size % self.d_model == 0
        ), "batch_size must be divisible by d_model"
        dl_batch_size = self.batch_size // self.seq_len
        self.dl = self.mk_dl(dl_batch_size)
        self._iter = iter(self.dl)

    def mk_dl(self, batch_size: int) -> t.utils.data.DataLoader:
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

        return t.utils.data.DataLoader(
            self.ds,  # type: ignore
            batch_size=batch_size,
            collate_fn=_collate_fn,
            drop_last=True,
            num_workers=10,
            prefetch_factor=10,
            pin_memory=False,
            shuffle=True,
        )

    def next(self) -> Float[t.Tensor, "batch n_models d_model"]:
        try:
            batch = next(self._iter)
            if self.cfg.normalization_factor is not None:
                batch *= self.cfg.normalization_factor[None, :, None]
                return batch

        except StopIteration:
            warnings.warn("Dataloader exhausted, resetting iterator")
            self._iter = iter(self.dl)
            return self.next()

    def __len__(self):
        return len(self.dl)

    def model_names(self):
        return list(self.ds.features.keys())


if __name__ == "__main__":
    cfg = MultiFeatureBufferConfig(
        activations_path="./activations/pythia-70m-layer-4-pile-resid-post-activations-through-time",
        hook_name="blocks.4.hook_resid_post",
        batch_size=4096,
    )
    buffer = MultiFeatureBuffer(cfg)
    for i in range(10):
        batch = buffer.next()
        print(batch.shape)
