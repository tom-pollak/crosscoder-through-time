from pathlib import Path
from tqdm import tqdm, trange
import torch as t
import wandb
from dataclasses import dataclass

from crosscoder.model import CrossCoder, CrossCoderConfig
from crosscoder.buffer import CachedBuffer, CachedBufferConfig
from crosscoder.buffer_on_the_fly import FlyBuffer, FlyBufferConfig


@dataclass
class TrainerConfig:
    # Training
    batch_size: int
    # Optimizer
    lr: float
    beta1: float
    beta2: float
    l1_coeff: float
    warmup_steps: int | None
    warmup_pct: float | None
    # Logging
    wandb_project: str
    wandb_entity: str
    log_every: int
    save_every: int
    dump_dir: str

    def __post_init__(self):
        assert self.warmup_pct is None or self.warmup_steps is None


class Trainer:
    def __init__(
        self,
        trainer_cfg: TrainerConfig,
        crosscoder_cfg: CrossCoderConfig,
        buffer_cfg: FlyBufferConfig | CachedBufferConfig,
    ):
        self.cfg = trainer_cfg
        self.crosscoder = CrossCoder(crosscoder_cfg)
        self.buffer = (
            CachedBuffer(buffer_cfg)
            if isinstance(buffer_cfg, CachedBufferConfig)
            else FlyBuffer(buffer_cfg)
        )
        self.step_counter = 0
        self.total_steps = len(self.buffer)
        self.version = self.create_version()

        self.root_save_dir = Path(self.cfg.dump_dir) / f"version_{self.version}"
        self.root_save_dir.mkdir(parents=True, exist_ok=True)

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        warmup_steps = (
            self.cfg.warmup_pct * self.total_steps
            if self.cfg.warmup_pct is not None
            else self.cfg.warmup_steps
        )
        assert warmup_steps is not None

        if self.step_counter < warmup_steps:
            return self.cfg.l1_coeff * self.step_counter / (warmup_steps)
        else:
            return self.cfg.l1_coeff

    def step(self, batch):
        losses = self.crosscoder.get_losses(batch)
        loss = losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            **{
                f"explained_variance_{nm}": losses.explained_variance_per_model[i]
                .mean()
                .item()
                for i, nm in enumerate(self.buffer.model_names())
            },
        }
        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(
            loss_dict,
            step=self.step_counter,
            commit=(self.step_counter + 1) % self.cfg.log_every == 0,
        )
        if (self.step_counter + 1) % self.cfg.log_every == 0:
            tqdm.write(str(loss_dict))

    def create_version(self) -> int:
        base_dir = Path(self.cfg.dump_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        version_list = [
            int(file.name.split("_")[1])
            for file in list(base_dir.iterdir())
            if "version" in str(file)
        ]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0

    # @property
    # def save_dir(self) -> Path:
    #     return self.root_save_dir / f"checkpoint_{self.step_counter}"

    # def save(self):
    #     self.save_dir.mkdir(parents=True, exist_ok=False)
    #     print("Saving CrossCoder")
    #     self.crosscoder.save(self.save_dir)
    #     print("Saving Buffer")
    #     self.buffer.save(self.save_dir)
    #     print("Saving Trainer")
    #     with open(f"{self.save_dir}/trainer_cfg.json", "w") as f:
    #         json.dump(asdict(self.cfg), f)
    #     state = {
    #         "step_counter": self.step_counter,
    #         "optimizer_state": self.optimizer.state_dict(),
    #         "scheduler_state": self.scheduler.state_dict(),
    #     }
    #     t.save(state, f"{self.save_dir}/trainer_state.pt")

    # @classmethod
    # def load(cls, dump_dir: Path, version: int, step: int):
    #     save_dir = dump_dir / f"version_{version}" / f"checkpoint_{step}"
    #     crosscoder = CrossCoder.load(save_dir)
    #     buffer = Buffer.load(save_dir)
    #     trainer_cfg = TrainerConfig(**json.load(open(f"{save_dir}/trainer_cfg.json")))
    #     print(f"TrainerConfig:\n{trainer_cfg}")
    #     trainer = cls(
    #         trainer_cfg,
    #         crosscoder.cfg,
    #         buffer.cfg,
    #         buffer._models_dict,
    #         buffer.tokens_dl,
    #     )
    #     trainer.crosscoder = crosscoder
    #     trainer.buffer = buffer
    #     trainer.step_counter = step
    #     trainer.optimizer.load_state_dict(t.load(f"{save_dir}/optimizer_state.pt"))
    #     trainer.scheduler.load_state_dict(t.load(f"{save_dir}/scheduler_state.pt"))
    #     gc.collect()
    #     return trainer

    def train(self):
        wandb.init(project=self.cfg.wandb_project, entity=self.cfg.wandb_entity)
        self.optimizer = t.optim.Adam(
            self.crosscoder.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        self.step_counter = 0
        try:
            for i in trange(self.total_steps, desc="Training"):
                batch = self.buffer.next()
                loss_dict = self.step(batch)
                self.log(loss_dict)
                # if (i + 1) % self.cfg.save_every == 0:
                #     self.save()
        except KeyboardInterrupt:
            print("Keyboard interrupt, saving checkpoint...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            pass
            # self.save()
