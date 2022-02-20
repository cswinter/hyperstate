from dataclasses import dataclass
import tempfile
import textwrap
import os
from typing import Any

import numpy as np

from hyperstate.hyperstate import HyperState
from hyperstate.lazy import Lazy, Serializable


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


class Params(Serializable):
    def __init__(self) -> None:
        self.params = np.zeros(64)

    def serialize(self) -> np.ndarray:
        return self.params

    @classmethod
    def deserialize(clz, state_dict: np.ndarray, config: Any, state: Any) -> "Params":
        result = Params()
        result.params = state_dict
        return result


@dataclass
class State(Lazy):
    step: int
    params: Params


class HS(HyperState[Config, State]):
    def __init__(self, initial_config: str, checkpoint_dir: str):
        super().__init__(Config, State, initial_config, checkpoint_dir)

    def initial_state(self) -> State:
        return State(step=0, params=Params())


def test_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write initial config file
        config = """\
        Config(
            lr: 0.01,
            steps: 100,
            ppo: PPO(
                cliprange: 0.3,
                gamma: 0.999,
                lambd: 0.95,
                entcoeff: "step: 0.1@0 0.0@100",
                value_loss_coeff: 2,
            )
        )
        """
        with open(tmpdir + "/config.ron", "w") as f:
            f.write(textwrap.dedent(config))
        checkpoint_dir = tmpdir + "/checkpoint"
        os.mkdir(checkpoint_dir)
        hs = HS(tmpdir + "/config.ron", checkpoint_dir)
        hs.state.step = 50
        hs.state.params.params += 0.1
        hs.step()

        # Restore from checkpoint
        hs2 = HS(tmpdir + "/config.ron", checkpoint_dir)
        assert hs2.state.step == 50
        assert (hs2.state.params.params == hs.state.params.params).all()
        assert hs2.config == Config(
            lr=0.01,
            steps=100,
            ppo=PPO(
                cliprange=0.3,
                gamma=0.999,
                lambd=0.95,
                entcoeff=0.05,
                value_loss_coeff=2,
            ),
        )


@dataclass(eq=True)
class MinConfig:
    lr: float
    steps: int


@dataclass
class MinState(Lazy):
    step: int


class MinHS(HyperState[MinConfig, MinState]):
    def __init__(self, initial_config: str, checkpoint_dir: str):
        super().__init__(
            MinConfig,
            MinState,
            initial_config,
            checkpoint_dir,
            ignore_extra_fields=True,
        )

    def initial_state(self) -> MinState:
        return MinState(step=0)


def test_ignore_extra_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write initial config file
        config = """\
        Config(
            lr: 0.01,
            steps: 100,
            ppo: PPO(
                cliprange: 0.3,
                gamma: 0.999,
                lambd: 0.95,
                entcoeff: "step: 0.1@0 0.0@100",
                value_loss_coeff: 2,
            )
        )
        """
        with open(tmpdir + "/config.ron", "w") as f:
            f.write(textwrap.dedent(config))
        checkpoint_dir = tmpdir + "/checkpoint"
        os.mkdir(checkpoint_dir)
        hs = HS(tmpdir + "/config.ron", checkpoint_dir)
        hs.state.step = 50
        hs.state.params.params += 0.1
        hs.step()

        # Restore from checkpoint
        MinHS(tmpdir + "/config.ron", checkpoint_dir)
