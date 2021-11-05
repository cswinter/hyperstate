# HyperState


Opinionated library for managing hyperparameter configs and mutable program state of machine learning training systems.

**Key Features**:
- (De)serialize nested Python dataclasses as [RON files](https://github.com/ron-rs/ron)
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

Configs can be deserialized with `hyperstate.load`.
The `load` method requires two arguments, the type/class of the config and a path to a config file:

```rust
// config.ron
Config(
    lr: 0.1,
    batch_size: 256,
)
```

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

Configs can be serialized with `hyperstate.dump`.
The second argument to `dump` is a path to a file, which can be omitted to return the serialized config as a string instead of saving it to a file:

```python
>>> print(hyperstate.dump(Config(lr=0.1, batch_size=256))
Config(
    lr: 0.1,
    batch_size: 256,
)
```

## State

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

## Serializable

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
