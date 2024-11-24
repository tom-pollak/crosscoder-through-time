import torch as t
from datasets import Dataset, load_dataset
from sae_lens.load_model import load_model
from buffer_on_the_fly import BufferConfig
from transformer_lens import HookedTransformer
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
    batch_size=4096,
    lr=5e-5,
    beta1=0.9,
    beta2=0.999,
    l1_coeff=2,
    # Dataset -- only for cached acts
    dataset_repo_id=dataset_repo_id,
    # Logging
    wandb_project="crosscoder-time",
    wandb_entity="tompollak",
    log_every=100,
    save_every=1_000,
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

buffer_cfg = BufferConfig(
    sae_batch_size=4096,
    model_batch_size=128,
    buffer_mult=256,
    seq_len=1024,
    hook_point="blocks.4.hook_resid_post",
    hook_layer=4,
    device=str(device),
)

steps = [
    1000,
    5000,
    10_000,
    50_000,
    100_000,
    143_000,
]

models: dict[int, HookedTransformer] = {  # type: ignore
    step: t.compile(
        load_model(
            model_class_name="HookedTransformer",
            model_name="EleutherAI/pythia-70m",
            device=str(device),
            model_from_pretrained_kwargs={"revision": f"step{step}"},
        )
    )
    for step in steps
}


ds = load_dataset("NeelNanda/pile-small-tokenized-2b", split="train")
assert isinstance(ds, Dataset)
ds = ds.with_format("torch")

tokens_dl = t.utils.data.DataLoader(
    ds,  # type: ignore
    batch_size=buffer_cfg.model_batch_size,
    shuffle=True,
    num_workers=10,
    prefetch_factor=10,
    pin_memory=True,
    # persistent_workers=True,
    drop_last=True,
)

if __name__ == "__main__":
    trainer = Trainer(trainer_cfg, crosscoder_cfg, buffer_cfg, models, tokens_dl)
    trainer.train()

    # Push to hub
    api = HfApi()
    api.create_repo(repo_id=model_repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=trainer.cfg.dump_dir,
        repo_id=model_repo_id,
        commit_message="Training finished",
    )
