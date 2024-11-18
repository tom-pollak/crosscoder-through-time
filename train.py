import os

os.environ["HF_HOME"] = "/home/jl_fs/"

import torch as t
from trainer import Trainer, TrainerConfig
from crosscoder import CrossCoderConfig
from huggingface_hub import HfApi

device = "cuda" if t.cuda.is_available() else "mps" if t.mps.is_available() else "cpu"

model_repo_id = "tommyp111/pythia-70m-crosscoder-through-time"  # to push to

dataset_repo_id = (
    "tommyp111/pythia-70m-layer-4-pile-resid-post-activations-through-time"
)

trainer_cfg = TrainerConfig(
    # Training
    batch_size=512, # 4096
    num_tokens=10_000_000,
    lr=5e-5,
    beta1=0.9,
    beta2=0.999,
    l1_coeff=2,
    # Dataset
    dataset_repo_id=dataset_repo_id,
    dataset_kwargs={},  # {"data_dir": local_data_dir},
    shuffle=True,
    seed=49,
    # Logging
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
    HfApi().upload_folder(
        folder_path=trainer.cfg.dump_dir,
        repo_id=model_repo_id,
        commit_message="Training finished",
    )
