import torch as t
from transformer_lens import HookedTransformer
from sae_lens.load_model import load_model
from huggingface_hub import HfApi

from crosscoder.buffer_on_the_fly import FlyBufferConfig
from crosscoder.buffer_cached import CachedBufferConfig
from crosscoder.model import CrossCoderConfig
from crosscoder.trainer import Trainer, TrainerConfig

device = "cuda" if t.cuda.is_available() else "mps" if t.mps.is_available() else "cpu"

model_repo_id = "tommyp111/pythia-70m-crosscoder-through-time"  # to push to

steps = [100000, 143000]

trainer_cfg = TrainerConfig(
    # Training
    batch_size=4096,
    lr=5e-5,
    beta1=0.9,
    beta2=0.999,
    l1_coeff=1.25,
    warmup_steps=5000,
    warmup_pct=None,
    # Logging
    wandb_project="crosscoder-time",
    wandb_entity="tompollak",
    log_every=100,
    save_every=25_000,
    dump_dir="./checkpoints",
)

crosscoder_cfg = CrossCoderConfig(
    d_in=512,
    dict_size=2**14,  # 32K
    n_models=len(steps),
    enc_dtype="float32",
    dec_init_norm=0.08,
    device=str(device),
    seed=49,
)


# models: dict[int, HookedTransformer] = {  # type: ignore
#     step: t.compile(
#         load_model(
#             model_class_name="HookedTransformer",
#             model_name="EleutherAI/pythia-70m",
#             device=str(device),
#             model_from_pretrained_kwargs={"revision": f"step{step}"},
#         )
#     )
#     for step in steps
# }
# buffer_cfg = BufferConfig(
#     models=models,
#     sae_batch_size=4096,
#     model_batch_size=128,
#     buffer_mult=511,  # gives a nice even number after stripping BOS
#     seq_len=1024,
#     hook_point="blocks.4.hook_resid_post",
#     hook_layer=4,
#     device=str(device),
# )

buffer_cfg = CachedBufferConfig(
    activations_path="./activations/pythia-70m-layer-4-pile-resid-post-activations-through-time",
    hook_name="blocks.4.hook_resid_post",
    batch_size=4096,
    model_names=[f"step{step}" for step in steps],
    seed=49,
    normalization_factor=t.tensor(
        [1.0066, 1.4880],
        # [0.6371, 0.6009, 1.0066, 1.4880],
        device="cpu",
        dtype=t.float32,
    ),
)


if __name__ == "__main__":
    trainer = Trainer(trainer_cfg, crosscoder_cfg, buffer_cfg)
    # trainer = Trainer.load(Path("./checkpoints"), version=0, step=25_000)
    trainer.train()

    final_model_checkpoint_dir = trainer.root_save_dir / "model_checkpoint_final"
    final_model_checkpoint_dir.mkdir(parents=True, exist_ok=False)
    trainer.crosscoder.save(final_model_checkpoint_dir)

    # Push to hub
    api = HfApi()
    api.create_repo(repo_id=model_repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=final_model_checkpoint_dir,
        repo_id=model_repo_id,
        commit_message="Training finished",
    )
