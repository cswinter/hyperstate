# HyperState


Opinionated library for managing hyperparameter configs and mutable program state of machine learning training systems.

**Key Features**:
- (De)serialize nested Python dataclasses as [Rusty Object Notation](https://github.com/ron-rs/ron)
- Override any config value from the command line
- Automatic checkpointing and restoration of full program state
- Checkpoints are (partially) human readable and can be modified in a text editor
- Powerful tools for versioning and schema evolution that can detect breaking changes and make it easy to restructure your program while remaining backwards compatible with old checkpoints
- Large binary objects in checkpoints can be loaded lazily only when accessed
- DSL for hyperparameter schedules 
- (planned) Edit hyperparameters of running experiments on the fly without restarts
- (planned) Usable without fermented vegetables

## Configs

HyperState supports a strictly typed subset of Python objects:
- dataclasses
- containers: `Dict`, `List`, `Tuple`, `Optional`
- primitives: `int`, `float`, `str`, `Enum`
- objects with custom serialization logic: [`hyperstate.Serializable`](#serializable)

Use `hyperstate.dump` to serialize configs.
The second argument to `dump` is a path to a file, and can be omitted to return the serialized config as a string instead of saving it to a file:

```python
>>> print(hyperstate.dump(Config(lr=0.1, batch_size=256))
Config(
    lr: 0.1,
    batch_size: 256,
)
```

Use `hyperstate.load` to deserialize configs.
The `load` method requires two arguments, the type/class of the config and a path to a config file:

```python
@dataclass
class Config:
    lr: float
    batch_size: int

config: Config = hyperstate.load(Config, "config.ron")
```

The `load` method also accepts an optional list of `overrides` that can be used to set the value of any config field:

```python
@dataclass
class OptimizerConfig:
    lr: float
    batch_size: int

@dataclass
class Config:
    optimzer: OptimizerConfig
    steps: int

overrides = ["optimizer.lr=0.1", "steps=100"]
config: Config = hyperstate.load(Config, "config.yaml", overrides=overrides)
```

## Versioning

Versioning allows you to modify your `Config` class while still remaining compatible with checkpoints recorded at previous version.
To benefit from versionining, your config must inherit `hyperstate.Versioned` and implement its `version` function:

```python
@dataclass
class Config(hyperstate.Versioned):
    lr: float
    batch_size: int
    
    @classmethod
    def version(clz) -> int:
        return 0
```

When serializing the config, hyperstate will now record an additional `version` field with the value of the current version.
Any snapshots that contain configs without a version field are assumed to have a version of `0`.

### `RewriteRule`

Now suppose you modify your `Config` class, e.g. by renaming the `lr` field to `learning_rate`.
To still be able to load old configs that are using `lr` instead of `learning_rate`, you increase the `version` to `1` and add an entry to the dictionary returned by `upgrade_rules` that tells HyperState to change `lr` to `learning_rate` when upgrading configs from version `0`.

```python
from dataclasses import dataclass
from typing import Dict, List
from hyperstate import Versioned
from hyperstate.schema.rewrite_rule import RenameField, RewriteRule

@dataclass
class Config(Versioned):
    learning_rate: float
    batch_size: int
    
    @classmethod
    def version(clz) -> int:
        return 1

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        """
        Returns a list of rewrite rules that can be applied to the given version
        to make it compatible with the next version.
        """
        return {
            0: [RenameField(old_field=("lr",), new_field=("learning_rate",))],
        }
```

In the majority of cases, you don't actually have to manually write out `RewriteRule`s.
Instead, they are generated for you automatically by the [Schema Evolution CLI](#schema-evolution-cli).

### Schema evolution CLI

HyperState comes with a command line tool for managing changes to your config schema.
To access the CLI, simply add the following code to the Python file defining your config:

```python
# config.py
from hyperstate import schema_evolution_cli

if __name__ == "__main__":
    schema_evolution_cli(Config)
```

Run `python config.py` to see a list of available commands, described in more detail below.

#### `dump-schema`

The `dump-schema` command creates a file describing the schema of your config.
This file should commited to version control, and is used to detect changes to the config schema and perform automatic upgrades.

#### `check-schema`

The `check-schema` command compares your config class to a schema file and detects any backwards incompatible changes.
It also emits a suggested list of [`RewriteRule`](#rewrite-rule)s that can upgrade old configs to the new schema.
HyperState does not always guess the correct `RewriteRule`s so you still need to check that they are correct.

```
$ python config.py check-schema
WARN  field renamed to learning_rate: lr
WARN  schema changed but version identical
Schema incompatible

Proposed mitigations
- add upgrade rules:
    0: [
        RenameField(old_field=('lr',), new_field=('learning_rate',)),
    ],
- bump version to 1
```

#### `upgrade-schema`

The `upgrade-schema` command functions much the same as `check-schema`, but also updates your schema config files once all backwards-incompatability issues have been address.

#### `upgrade-config`

The `upgrade-config` command takes a list of paths to config files, and upgrades them to the latest version.

### Automated Tests

To prevent accidental backwards-incompatible modifications of your `Config` class, you can use the following code as an automated test that checks your config `Class` against a schema file created with [`dump-schema`](#dump-schema): 

```python
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import load_schema
from config import Config

def test_schema():
    old = load_schema("config-schema.ron")
    checker = SchemaChecker(old, Config)
    if checker.severity() >= Severity.WARN:
        print(checker.print_report())
    assert checker.severity() == Severity.INFO
```

## `Serializable`


```python
from dataclass import @dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperstate

@dataclass
class Config:
   inputs: int

# To define custom serialization logic for a class, inherit from `hyperstate.Serializable` and implementing `serialize` and `deserialize`.
class LinearRegression(nn.Module, hyperstate.Serializable):
    def __init__(self, inputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputs, 1)
        
    def forward(self, x):
        return self.fc1(x)
    
    # `serialize` should return an object composed only of dicts, lists, primitive types, numpy arrays and PyTorch tensors
    def serialize(self) -> Any:
        return self.state_dict()

    @classmethod
    def deserialize(clz, state_dict, ctx):
        net = clz(ctx["config"].inputs)
        return net.load_state_dict(state_dict)


@dataclass
class State:
    net: LinearRegression

config = hyperstate.load("config.ron")
state = hyperstate.load("state.ron", ctx={"config": config})
```

- interface
- ctx

## `Lazy`

## `HyperState`

State objects must also be `@dataclass`es, and can additonally include opaque `hyperstate.Blob[T]` types with custom (de)serialization logic.
Both `Config` and `State` are managed by a `HyperState[Config, State]` object with `config` and `state` fields.
The `HyperState` object is created/loaded with `HyperState.load`:

```python
def load(
    config_clz: Type[C],
    state_clz: Type[S],
    initial_state: Callable[[C], S],
    path: str,
    checkpoint_dir: Optional[str] = None,
    checkpoint_key: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> "HyperState[C, S]":
    """
    Loads a HyperState from a checkpoint (if exists) or initializes a new one.

    :param config_clz: The type of the config object.
    :param state_clz: The type of the state object.
    :param initial_state: A function that takes a config object and returns an initial state object.
    :param path: The path to the config file or full checkpoint directory.
    :param checkpoint_dir: The directory to store checkpoints.
    :param checkpoint_key: The key to use for the checkpoint. This must be a field of the state object (e.g. a field holding current iteration).
    :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
    """
    pass
```

### Checkpointing 

Just call `step` on the `HyperState` object to checkpoint the current config/state to the configured directory.

### Schedules

All `int`/`float` fields in the config can also be set to a schedule that will be updated at each step.

### Limitations

Currently, can't easily use `HyperState` with `PyTorch`.
Problem: To initialize optimizer class, needs to have parameter state as well as config.
Therefore, `HyperState` currently only stores the optimizer state and policy state which has to be manually updated each step.

Sketch of solution:
- the `State` dataclasses have to inherit/add property which intercepts fields accesses and gives transparent lazy loading
- therefore, we can partially initialize the state object and pass `State` to all init calls as well
- as long as there are no loops, initializer can access any other (fully initialized) state objects
- this allows us to support any types that implement a some hyperstate serialization interface (load(config, state, state_dict) -> self, get_state_dict() -> state_dict)

Limitation 2: Syncing state in distributed training.
