from dataclasses import dataclass
from typing import Any, Literal
from .help import help
import click


@dataclass
class BaseOptimizerConfig:
    """Optimizer hyperparameters

    Attributes:
        lr: learning rate
        anneal_lr: anneal learning rate
        max_grad_norm: max gradient norm
    """

    lr: float = 1e-4
    anneal_lr: bool = True
    max_grad_norm: float = 100.0


@dataclass
class OptimizerConfig(BaseOptimizerConfig):
    """Optimizer hyperparameters

    Attributes:
        batch_size: batch size
    """

    batch_size: int = 512


@dataclass
class WandbConfig:
    """W&B tracking settings.

    Attributes:
        track: whether to track metrics to W&B
        project_name: the wandb's project name
        entity: the entity (team) of wandb's project
    """

    track: bool = False
    project_name: str = "enn-bc"
    entity: str = "entity-neural-network"


@dataclass
class Config:
    """Supervised training configuration.

    Attributes:
        filepath: filepath to load dataset from
        epochs: number of epochs to train for
        loss_fn: loss function ("kl" or "mse")
        log_interval: print out loss every log_interval steps
        fast_eval_interval: interval at which to evaluate with subset of test data
        fast_eval_samples: number of samples to use in fast evaluation
    """

    optim: OptimizerConfig
    wandb: WandbConfig
    filepath: str
    epochs: int = 10
    loss_fn: Literal["kl", "mse"] = "mse"
    log_interval: int = 10
    fast_eval_interval: int = 32768
    fast_eval_samples: int = 8192


def test_help(capsys: Any) -> None:
    help(Config)
    captured = capsys.readouterr()
    assert (
        click.unstyle(captured.out)
        == """optim: OptimizerConfig
  lr: float = 0.0001                      # learning rate
  anneal_lr: bool = True                  # anneal learning rate
  max_grad_norm: float = 100.0            # max gradient norm
  batch_size: int = 512                   # batch size
wandb: WandbConfig
  track: bool = False                     # whether to track metrics to W&B
  project_name: str = 'enn-bc'            # the wandb's project name
  entity: str = 'entity-neural-network'   # the entity (team) of wandb's project
filepath: str                             # filepath to load dataset from
epochs: int = 10                          # number of epochs to train for
loss_fn: 'kl'|'mse' = 'mse'               # loss function ("kl" or "mse")
log_interval: int = 10                    # print out loss every log_interval steps
fast_eval_interval: int = 32768           # interval at which to evaluate with subset of test data
fast_eval_samples: int = 8192             # number of samples to use in fast evaluation
"""
    )
