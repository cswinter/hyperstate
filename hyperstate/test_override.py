from dataclasses import dataclass
import hyperstate as hs


@dataclass(eq=True)
class PPO:
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
        path=None,
        overrides=[
            "task_id=CherryPick",
            "lr=0.1",
            "steps=100",
            "ppo.cliprange=0.1",
        ],
    )

    assert config == Config(
        lr=0.1,
        steps=100,
        ppo=PPO(cliprange=0.1),
        task_id="CherryPick",
    )
