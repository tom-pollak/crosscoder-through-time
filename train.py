import torch as t
from trainer import Trainer, TrainerConfig
from crosscoder import CrossCoderConfig

device = "cuda" if t.cuda.is_available() else "mps" if t.mps.is_available() else "cpu"

repo_id = "tommyp111/pythia-70m-layer-4-pile-resid-post-activations-through-time"

trainer_cfg = TrainerConfig(
    batch_size=4096,
    num_tokens=10_000_000,
    lr=5e-5,
    beta1=0.9,
    beta2=0.999,
    l1_coeff=2,
    repo_id=repo_id,
    seed=49,
    wandb_project="crosscoder-time",
    wandb_entity="tompollak",
    log_every=100,
    save_every=500,
    dump_dir="./checkpoints",
)

crosscoder_cfg = CrossCoderConfig(
    d_in=512,
    dict_size=2**14,
    n_models=6,
    enc_dtype="float32",
    dec_init_norm=0.08,
    device=str(device),
    seed=49,
)

if __name__ == "__main__":
    trainer = Trainer(trainer_cfg, crosscoder_cfg)
    trainer.train()
