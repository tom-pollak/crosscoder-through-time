import json
from crosscoder import CrossCoder, CrossCoderConfig
from buffer import MultiFeatureBuffer
import tqdm
import torch as t
import wandb
from dataclasses import asdict, dataclass


@dataclass
class TrainerConfig:
    # Training
    batch_size: int
    num_tokens: int
    # Optimizer
    lr: float
    beta1: float
    beta2: float
    l1_coeff: float
    # Dataset
    dataset_repo_id: str
    seed: int
    # Logging
    wandb_project: str
    wandb_entity: str
    log_every: int
    save_every: int
    dump_dir: str


class Trainer:
    def __init__(self, trainer_cfg: TrainerConfig, crosscoder_cfg: CrossCoderConfig):
        self.cfg = trainer_cfg
        self.crosscoder = CrossCoder(crosscoder_cfg).to(crosscoder_cfg.device)
        self.buffer = MultiFeatureBuffer(self.cfg.dataset_repo_id, seed=self.cfg.seed)
        self.dl = self.buffer.iter(batch_size=self.cfg.batch_size)

        self.total_steps = self.cfg.num_tokens // self.cfg.batch_size

        self.optimizer = t.optim.Adam(
            self.crosscoder.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.beta1, self.cfg.beta2),
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        self.step_counter = 0

        wandb.init(project=self.cfg.wandb_project, entity=self.cfg.wandb_entity)

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        # Linearly increases from 0 to cfg["l1_coeff"] over the first 0.05 * self.total_steps steps, then keeps it constant
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg.l1_coeff * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg.l1_coeff

    def step(self):
        acts = next(self.dl)
        losses = self.crosscoder.get_losses(acts)
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
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        self.step_counter += 1
        return loss_dict

    def log(self, loss_dict):
        wandb.log(loss_dict, step=self.step_counter)
        print(loss_dict)

    def save(self):
        self.crosscoder.save(self.cfg.dump_dir)
        with open(f"{self.cfg.dump_dir}/trainer_cfg.json", "w") as f:
            json.dump(asdict(self.cfg), f)

    def train(self):
        self.step_counter = 0
        try:
            for i in tqdm.trange(self.total_steps):
                loss_dict = self.step()
                if i % self.cfg.log_every == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg.save_every == 0:
                    self.save()
        finally:
            self.save()
