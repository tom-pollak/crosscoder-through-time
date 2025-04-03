# %%
from torch import nn
import torch as t
import einops


class AutoEncoderTopK(nn.Module):
    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x: t.Tensor, return_topk: bool = False):
        print(x.shape)
        print(self.b_dec.shape)
        print(self.encoder.weight.shape)
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.weight.dtype).eps
        norm = t.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.weight.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.weight.grad,
            self.decoder.weight.data,
            "d_in d_sae, d_in d_sae -> d_sae",
        )
        self.decoder.weight.grad -= einops.einsum(
            parallel_component,
            self.decoder.weight.data,
            "d_sae, d_in d_sae -> d_in d_sae",
        )


def loss(x, ae: AutoEncoderTopK, step=None, logging=False):
    f = ae.encode(x)
    x_hat = ae.decode(f)
    e = x_hat - x
    loss = e.pow(2).sum(dim=-1).mean()
    return loss


# %%

from ezmup import Ezmup, get_coord_data, plot_coord_data

model = AutoEncoderTopK(activation_dim=47, dict_size=10, k=10)
mup_engine = Ezmup(47, model, init_std=1.0)
mup_engine.change_width_as(64)

# %%

# model(t.randn(16, 47))

# %%

mup_engine.forward = loss

# example run
x = t.randn(16, 47)

df = get_coord_data(mup_engine, x, n_seeds=1, n_steps=3)
df.to_csv("contents/example.csv")


plot_coord_data(
    df,
    y="l1",
    save_to="contents/coord-check.png",
    suptitle=None,
    x="width",
    hue="module",
)

# %%
