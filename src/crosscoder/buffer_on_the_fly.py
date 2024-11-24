from datasets import Dataset, load_dataset
from typing import Any
import numpy as np
import einops
import torch as t
from jaxtyping import Float
from transformer_lens import HookedTransformer
import tqdm
from dataclasses import dataclass
import math


@dataclass
class BufferConfig:
    token_dataset_repo_id: str
    models: dict[Any, HookedTransformer]
    buffer_mult: int
    model_batch_size: int
    sae_batch_size: int
    seq_len: int
    hook_point: str
    hook_layer: int
    device: str


class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(
        self,
        cfg: BufferConfig,
    ):
        ds = load_dataset(cfg.token_dataset_repo_id, split="train")
        assert isinstance(ds, Dataset)
        ds = ds.with_format("torch")

        self.tokens_dl = t.utils.data.DataLoader(
            ds,  # type: ignore
            batch_size=cfg.model_batch_size,
            shuffle=True,
            num_workers=10,
            prefetch_factor=10,
            pin_memory=True,
            # persistent_workers=True,
            drop_last=True,
        )
        self._iter = iter(self.tokens_dl)

        self.cfg = cfg
        # 1023 * 4096 / (1024 - 1) = 4096
        # X * 4.003910068426
        num_sequences_needed = math.floor(
            cfg.buffer_mult * cfg.sae_batch_size / (cfg.seq_len - 1)
        )
        self.buffer_size = num_sequences_needed * (cfg.seq_len - 1)
        # 4100 / 128 = 32.03
        self.refresh_batches = math.ceil(num_sequences_needed / cfg.model_batch_size)
        self.total_batches = (
            len(self.tokens_dl) // self.refresh_batches
        ) * cfg.buffer_mult

        print("=== Buffer Config ===")
        print(f"Buffer size: {self.buffer_size}")
        print(f"Refresh batches: {self.refresh_batches}")
        print(f"Total batches: {self.total_batches}")
        print()

        self.buffer = t.zeros(
            (self.buffer_size, len(self.models), self.models[0].cfg.d_model),
            dtype=t.bfloat16,
            requires_grad=False,
            device="cpu",
        )
        self.pointer = 0
        self.normalize = True

        # self.normalisation_factor = t.tensor(
        #     [
        #         self.estimate_norm_scaling_factor(model)
        #         for model in tqdm.tqdm(models, leave=False)
        #     ],
        #     device=self.buffer.device,
        #     dtype=self.buffer.dtype,
        # )
        # print(f"Normalisation factor:\n{self.normalisation_factor}")
        self.normalisation_factor = t.tensor(
            [1.6593, 0.7984, 0.6371, 0.6009, 1.0066, 1.4880],
            device=self.buffer.device,
            dtype=self.buffer.dtype,
        )

        self.refresh()

    @property
    def models(self):
        return list(self.cfg.models.values())

    @t.no_grad()
    def estimate_norm_scaling_factor(
        self, model, n_batches_for_norm_estimate: int = 100
    ) -> int:
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for _, batch in zip(
            tqdm.trange(
                n_batches_for_norm_estimate,
                desc="Estimating norm scaling factor",
                leave=False,
            ),
            self.tokens_dl,
        ):
            tokens = batch["tokens"].to(self.cfg.device)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[self.cfg.hook_point],
                stop_at_layer=self.cfg.hook_layer + 1,
                return_type=None,
            )
            acts = cache[self.cfg.hook_point]
            # TODO: maybe drop BOS here
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm
        return scaling_factor

    @t.no_grad()
    def refresh(self):
        idx = 0
        with t.autocast("cuda", t.bfloat16):
            for _, batch in zip(
                tqdm.trange(
                    self.refresh_batches, desc="Refreshing buffer", leave=False
                ),
                self._iter,
            ):
                tokens = batch["tokens"].to(self.cfg.device)
                acts = []
                for model in self.models:
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=[self.cfg.hook_point],
                        stop_at_layer=self.cfg.hook_layer + 1,
                        return_type=None,
                    )
                    acts.append(cache[self.cfg.hook_point].cpu())
                acts = t.stack(acts)[:, :, 1:, :]  # Drop BOS
                assert acts.shape == (
                    len(self.models),
                    tokens.shape[0],
                    tokens.shape[1] - 1,
                    self.models[0].cfg.d_model,
                )  # [model, batch, seq_len, d_model]
                acts = einops.rearrange(
                    acts,
                    "model batch seq_len d_model -> (batch seq_len) model d_model",
                )
                self.buffer[idx : idx + acts.shape[0]] = acts[: self.buffer_size - idx]
                idx += acts.shape[0]
        self.buffer = self.buffer[t.randperm(self.buffer.shape[0])]

    @t.no_grad()
    def next(self) -> Float[t.Tensor, "batch model d_mosdel"]:
        # out: [batch_size, model, d_model]
        start = self.pointer
        end = start + self.cfg.sae_batch_size

        if end > self.buffer.shape[0]:
            self.refresh()
            start = 0
            end = self.cfg.sae_batch_size

        out = self.buffer[start:end].float()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]

        self.pointer = end
        return out

    def __len__(self):
        return self.total_batches

    def model_names(self):
        return list(self.cfg.models.keys())
