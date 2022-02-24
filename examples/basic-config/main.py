from dataclasses import dataclass
from hyperstate import command


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
    :param lr: Learning rate.
    :param batch_size: Batch size.
    """

    lr: float = 0.003
    batch_size: int = 512


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

    optimizer: OptimizerConfig
    net: NetConfig
    steps: int = 100


@command(Config)
def main(cfg: Config) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
