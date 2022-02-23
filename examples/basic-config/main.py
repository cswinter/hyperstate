from dataclasses import dataclass
import argparse
import hyperstate
from hyperstate.hyperstate import FieldNotFoundError
import click


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter values")
    parser.add_argument(
        "--hps-help", nargs="?", const="", help="Print help for hyperparameters"
    )
    args = parser.parse_args()
    if args.hps_help is not None:
        hyperstate.help(Config, args.hps_help)
    else:
        try:
            config = hyperstate.load(Config, file=args.config, overrides=args.hps)
            print(config)
        except FieldNotFoundError as e:
            print(click.style("ERROR", fg="red") + ": " + str(e))
            print()
            print("Most similar fields:")
            hyperstate.help(Config, e.field_name)
