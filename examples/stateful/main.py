from dataclasses import dataclass
import hyperstate
from hyperstate.hyperstate import StateManager


@dataclass
class NetConfig:
    """Configuration for the network.

    Attributes:
    :param hidden_size: Number of activations in the hidden layer.
    :param num_layers: Number of layers.
    """

    hidden_size: int = 128
    num_layers: int = 2


@dataclass
class Config:
    """Hyperparameter configuration.

    Attributes:
    :param optimizer: Optimizer configuration.
    :param net: Network configuration.
    :param steps: Number of steps to run.
    """

    net: NetConfig
    steps: int = 10


@dataclass
class State:
    step: int = 0


def initial_state(cfg: Config) -> State:
    return State()


@hyperstate.stateful_command(Config, State, initial_state)
def main(sm: StateManager) -> None:
    print(sm.config)
    print(sm.state)
    sm.state.step += 1
    sm.step()


if __name__ == "__main__":
    main()

# poetry run python examples/stateful/main.py --checkpoint-dir=.
