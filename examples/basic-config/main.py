from dataclasses import dataclass
import argparse
import hyperstate


@dataclass
class OptimizerConfig:
    lr: float = 0.003
    batch_size: int = 512


@dataclass
class NetConfig:
    hidden_size: int = 128
    num_layers: int = 2


@dataclass
class Config:
    optimizer: OptimizerConfig
    net: NetConfig
    steps: int = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--hps", nargs="+", help="Override hyperparameter value")
    args = parser.parse_args()
    config = hyperstate.load(Config, path=args.config, overrides=args.hps)
    print(config)
