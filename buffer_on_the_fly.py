import numpy as np
import einops
import torch as t
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer
import tqdm
from dataclasses import dataclass


@dataclass
class BufferConfig:
    buffer_mult: int
    batch_size: int
    seq_len: int
    hook_point: str
    hook_layer: int
    device: str


class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, cfg: BufferConfig, models: list[HookedTransformer], all_tokens: Float[t.Tensor, "batch seq_len"]):
        self.cfg = cfg
        self.buffer_size = cfg.batch_size * cfg.buffer_mult
        self.buffer_batches = self.buffer_size // (cfg.seq_len - 1)
        self.buffer_size = self.buffer_batches * (cfg.seq_len - 1)
        self.buffer = t.zeros(
            (self.buffer_size, len(models), models[0].cfg.d_model),
            dtype=t.bfloat16,
            requires_grad=False,
        ).to(cfg.device)  # hardcoding 2 for model diffing
        self.models = models
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens

        self.normalisation_factor = t.tensor(
            [
                self.estimate_norm_scaling_factor(cfg.batch_size, model)
                for model in models
            ],
            device=cfg.device,
            dtype=t.float32,
        )

        self.refresh()

    @t.no_grad()
    def estimate_norm_scaling_factor(
        self, batch_size, model, n_batches_for_norm_estimate: int = 100
    ) -> int:
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in tqdm.tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
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
        self.pointer = 0
        print("Refreshing the buffer!")
        with t.autocast("cuda", t.bfloat16):
            if self.first:
                num_batches = self.buffer_batches
            else:
                num_batches = self.buffer_batches // 2
            self.first = False
            for _ in tqdm.trange(0, num_batches, self.cfg.batch_size):
                tokens = self.all_tokens[
                    self.token_pointer : min(
                        self.token_pointer + self.cfg.batch_size, num_batches
                    )
                ].to(self.cfg.device)
                acts = []
                for model in self.models:
                    _, cache = model.run_with_cache(
                        tokens, names_filter=self.cfg.hook_point
                    )
                    acts.append(cache[self.cfg.hook_point])
                acts = t.stack(acts, dim=0)[:, :, 1:, :]  # Drop BOS
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

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg.batch_size

        self.pointer = 0
        self.buffer = self.buffer[t.randperm(self.buffer.shape[0]).to(self.cfg.device)]

    @t.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg.batch_size].float()
        # out: [batch_size, model, d_model]
        self.pointer += self.cfg.batch_size
        if self.pointer > self.buffer.shape[0] // 2 - self.cfg.batch_size:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
