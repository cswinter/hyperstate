from dataclasses import dataclass
from typing import Optional
import hyperstate as hs


@dataclass(eq=True)
class DeepInner:
    x: int


@dataclass(eq=True)
class PPO:
    inner: Optional[DeepInner] = None
    cliprange: float = 0.2
    gamma: float = 0.99
    lambd: float = 0.95
    entcoeff: float = 0.01
    value_loss_coeff: float = 1


@dataclass(eq=True)
class Config:
    lr: float
    steps: int
    ppo: PPO
    task_id: str


def test_override() -> None:
    config = hs.load(
        Config,
        file=None,
        overrides=[
            "task_id=CherryPick",
            "lr=0.1",
            "steps=100",
            "ppo.cliprange=0.1",
            "ppo.inner.x=10",
        ],
    )

    assert config == Config(
        lr=0.1,
        steps=100,
        ppo=PPO(cliprange=0.1, inner=DeepInner(x=10)),
        task_id="CherryPick",
    )
