from dataclasses import dataclass
import hyperstate


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


class Trainer(hyperstate.HyperState[Config, State]):
    def initial_state(cls) -> State:
        return State()

    def take_step(self) -> None:
        self.state.step += 1
        self.step()


@hyperstate.stateful_command(Trainer, Config, State)
def main(trainer: Trainer) -> None:
    print(trainer.config)
    print(trainer.state)
    trainer.take_step()


if __name__ == "__main__":
    main()

# poetry run python examples/stateful/main.py --checkpoint-dir=.
