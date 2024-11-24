"""
CrossCoder model from ckkissane/crosscoder-model-diff-replication
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, asdict
from jaxtyping import Float

from sae_lens.config import DTYPE_MAP

import torch as t
import einops
from huggingface_hub import hf_hub_download


@dataclass
class LossOutput:
    # loss: t.Tensor
    l2_loss: t.Tensor
    l1_loss: t.Tensor
    l0_loss: t.Tensor
    explained_variance: t.Tensor
    explained_variance_per_model: t.Tensor


@dataclass
class CrossCoderConfig:
    # model params
    d_in: int
    dict_size: int
    n_models: int
    # training params
    enc_dtype: str
    dec_init_norm: float
    # device params
    device: str
    seed: int


class CrossCoder(t.nn.Module):
    def __init__(self, cfg: CrossCoderConfig):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg.dict_size
        d_in = cfg.d_in
        self.dtype = DTYPE_MAP[cfg.enc_dtype]
        t.manual_seed(cfg.seed)
        self.W_enc = t.nn.Parameter(
            t.empty(cfg.n_models, d_in, d_hidden, dtype=self.dtype)
        )
        self.W_dec = t.nn.Parameter(
            t.nn.init.normal_(t.empty(d_hidden, cfg.n_models, d_in, dtype=self.dtype))
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data
            / self.W_dec.data.norm(dim=-1, keepdim=True)
            * cfg.dec_init_norm
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = t.nn.Parameter(t.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = t.nn.Parameter(t.zeros((cfg.n_models, d_in), dtype=self.dtype))
        self.d_hidden = d_hidden

        self.save_dir = None
        self.to(cfg.device)

    def encode(
        self, x: Float[t.Tensor, "batch n_models d_model"], apply_relu: bool = True
    ) -> Float[t.Tensor, "batch d_hidden"]:
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = t.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(
        self, acts: Float[t.Tensor, "batch d_hidden"]
    ) -> Float[t.Tensor, "batch n_models d_model"]:
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(
        self, x: Float[t.Tensor, "batch n_models d_model"]
    ) -> Float[t.Tensor, "batch n_models d_model"]:
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x: Float[t.Tensor, "batch n_models d_model"]) -> LossOutput:
        x = x.to(self.device).to(self.dtype)
        acts = self.encode(x)  # (batch, d_hidden)
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(
            squared_diff, "batch n_models d_model -> batch", "sum"
        )
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce(
            (x - x.mean(0)).pow(2), "batch n_models d_model -> batch", "sum"
        )
        explained_variance = 1 - l2_per_batch / total_variance

        explained_variances_per_model = []
        for i in range(self.cfg.n_models):
            per_token_l2_loss = (
                (x_reconstruct[:, i, :] - x[:, i, :]).pow(2).sum(dim=-1).squeeze()
            )
            total_variance = (x[:, i, :] - x[:, i, :].mean(0)).pow(2).sum(-1).squeeze()
            explained_variance_model = 1 - per_token_l2_loss / total_variance
            explained_variances_per_model.append(explained_variance_model)
        explained_variances_per_model = t.stack(explained_variances_per_model)

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(
            decoder_norms, "d_hidden n_models -> d_hidden", "sum"
        )
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts > 0).float().sum(-1).mean()

        return LossOutput(
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            l0_loss=l0_loss,
            explained_variance=explained_variance,
            explained_variance_per_model=explained_variances_per_model,
        )

    def save(self, save_dir: Path):
        weight_path = save_dir / "cc_weights.pt"
        cfg_path = save_dir / "cc_cfg.json"
        t.save(self.state_dict(), weight_path)
        with open(cfg_path, "w") as f:
            json.dump(asdict(self.cfg), f)

    @classmethod
    def load(cls, save_dir: Path):
        with open(f"{save_dir}/cc_cfg.json", "r") as f:
            cfg = CrossCoderConfig(**json.load(f))

        print(f"CrossCoderConfig:\n{cfg}")
        self = cls(cfg=cfg)
        self.load_state_dict(t.load(save_dir / "cc_weights.pt", map_location="cpu"))
        return self

    @property
    def device(self):
        return self.cfg.device


if __name__ == "__main__":
    cfg = CrossCoderConfig(
        d_in=512,
        dict_size=2**14,
        n_models=6,
        enc_dtype="float32",
        dec_init_norm=0.08,
        device="cuda",
        seed=49,
    )
    print(cfg)
