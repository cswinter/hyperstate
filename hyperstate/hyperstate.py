import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
from typing import (
    Generic,
    List,
    Any,
    Optional,
    Type,
    TypeVar,
    Dict,
    Tuple,
    Union,
)
from dataclasses import MISSING, dataclass
from hyperstate.schema.types import Struct, materialize_type

from hyperstate.schema.versioned import (
    VersionedSerializer,
    VersionedDeserializer,
)
from hyperstate.serde import (
    Deserializer,
    Serializer,
    asdict,
)
import hyperstate.serde as serde
from .lazy import LazyDeserializer, LazySerializer

import pyron

from hyperstate.schedule import Schedule, _parse_schedule

C = TypeVar("C")
S = TypeVar("S")
T = TypeVar("T")


class HyperState(ABC, Generic[C, S]):
    def __init__(
        self,
        config_clz: Type[C],
        state_clz: Type[S],
        initial_config: Union[str, Path],
        checkpoint_dir: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
        ignore_extra_fields: bool = False,
    ) -> None:
        """
        :param config_clz: The type of the config object.
        :param state_clz: The type of the state object.
        :param initial_config: Path to a config file or checkpoint.
        :param checkpoint_dir: Directory to store checkpoints. If the directory contains a valid checkpoint, the latest checkpoint will be loaded and `initial_config` will be ignored.
        :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
        :param ignore_extra_fields: If `True`, ignore extra fields in the config file.
        """
        self.config_clz = config_clz
        self.state_clz = state_clz
        self._last_checkpoint: Optional[Path] = None
        if isinstance(initial_config, str):
            initial_config = Path(initial_config)
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint = None
        if checkpoint_dir is not None:
            self.checkpoint_dir: Optional[Path] = Path(checkpoint_dir)
            checkpoint = find_latest_checkpoint(checkpoint_dir)
            if checkpoint is not None:
                print(f"Resuming from checkpoint {checkpoint}")
                initial_config = checkpoint
        else:
            self.checkpoint_dir = None

        if os.path.isdir(initial_config):
            config_path = initial_config / "config.ron"
            state_path: Optional[Path] = initial_config / "state.ron"
        else:
            config_path = initial_config
            state_path = None

        try:
            self.config, self.schedules = _typed_load(
                config_clz,
                file=config_path,
                overrides=overrides or [],
                allow_missing_version=state_path is not None,
                ignore_extra_fields=ignore_extra_fields,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e
        if state_path is None:
            self.state = self.initial_state()
        else:
            try:
                self.state = _typed_load(
                    state_clz,
                    file=state_path,
                    config=self.config,
                    ignore_extra_fields=ignore_extra_fields,
                )[0]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load state from {state_path}: {e}"
                ) from e
        _apply_schedules(self.state, self.config, self.schedules)

    @abstractmethod
    def initial_state(self) -> S:
        pass

    def checkpoint_key(self) -> str:
        return "step"

    def checkpoint(self, target_dir: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "checkpoint"
            p.mkdir()
            _typed_dump(self.config, p / "config.ron", self.schedules)
            _typed_dump(self.state, p / "state.ron")
            shutil.move(str(p), target_dir)

    def step(self) -> None:
        _apply_schedules(self.state, self.config, self.schedules)
        if self.checkpoint_dir is not None:
            val = getattr(self.state, self.checkpoint_key())
            assert isinstance(
                val, int
            ), f"checkpoint key `{self.checkpoint_key()}` must be an integer, but found value `{val}` of type `{type(val)}`"
            checkpoint_dir = (
                self.checkpoint_dir / f"latest-{self.checkpoint_key()}{val:012}"
            )
            self.checkpoint(str(checkpoint_dir))
            if self._last_checkpoint is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.move(str(self._last_checkpoint), tmpdir)
            self._last_checkpoint = checkpoint_dir
            # TODO: persistent checkpoints

    def config_dict(self) -> Any:
        return asdict(self.config, serializers=[VersionedSerializer()])


class FieldNotFoundError(Exception):
    def __init__(self, msg: str, field_name: str) -> None:
        super().__init__(msg)
        self.field_name = field_name


def _apply_schedules(state: Any, config: Any, schedules: Dict[str, Any]) -> None:
    for field_name, schedule in schedules.items():
        if isinstance(schedule, Schedule):
            schedule.update_value(config, state)
        else:
            assert isinstance(schedule, dict)
            _apply_schedules(state, getattr(config, field_name), schedule)


def _typed_dump(
    obj: Any,
    file: Union[Path, None] = None,
    schedules: Optional[Dict[str, Any]] = None,
    elide_defaults: bool = False,
) -> str:
    serializers = []
    lazy_serializer = LazySerializer()
    serializers = [lazy_serializer, VersionedSerializer()]
    if schedules is not None:
        serializers.append(ScheduleSerializer(schedules))
    if elide_defaults:
        serializers.append(ElideDefaults())
    if file is None:
        return serde.dumps(obj, serializers=serializers)
    else:
        result = serde.dump(obj, file, serializers=serializers)
        for blobpath, blob in lazy_serializer.blobs.items():
            with open(file.parent / blobpath, "wb") as f:
                f.write(blob)
        return result


def dump(obj: Any, file: Union[Path, str], elide_defaults: bool = False) -> None:
    if isinstance(file, str):
        file = Path(file)
    _typed_dump(obj, file, elide_defaults=elide_defaults)


def dumps(obj: Any, elide_defaults: bool = False) -> str:
    return _typed_dump(obj, elide_defaults=elide_defaults)


def _typed_load(
    clz: Type[T],
    *,
    file: Optional[Path] = None,
    data: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    config: Optional[Any] = None,
    allow_missing_version: bool = False,
    ignore_extra_fields: bool = False,
) -> Tuple[T, Dict[str, Any]]:
    if overrides is not None:
        deserializers: List[Deserializer] = [OverridesDeserializer(overrides)]
        schema = materialize_type(clz)
        assert isinstance(schema, Struct)
        for override in overrides:
            keyval = override.split("=", maxsplit=1)
            if len(keyval) == 1:
                raise ValueError(
                    f"Invalid override: {override}. Expected format: field.name=value"
                )
            path = keyval[0].split(".")
            if schema.find_field(path) is None:
                raise FieldNotFoundError(
                    f"Unknown field `{keyval[0]}` for class `{clz.__name__}`", path[-1]
                )
    else:
        deserializers = []
    schedules = ScheduleDeserializer()
    deserializers.append(schedules)
    deserializers.append(VersionedDeserializer(allow_missing_version))
    lazy = None
    assert file is None or data is None, "cannot specify both file and source"
    if file is not None:
        lazy = LazyDeserializer(config, file.absolute().parent)
        deserializers.append(lazy)
        value = serde.load(
            clz,
            file,
            deserializers=deserializers,
            ignore_extra_fields=ignore_extra_fields,
        )
    else:
        value = serde.loads(
            clz,
            data or "{}",
            deserializers=deserializers,
            ignore_extra_fields=ignore_extra_fields,
        )
    if lazy is not None and len(lazy.lazy_fields) > 0:
        value._unloaded_lazy_fields = lazy.lazy_fields  # type: ignore
    return value, schedules.schedules


def loads(
    clz: Type[T],
    value: str,
    overrides: Optional[List[str]] = None,
    ignore_extra_fields: bool = False,
) -> T:
    return _typed_load(
        clz, data=value, overrides=overrides, ignore_extra_fields=ignore_extra_fields
    )[0]


def load(
    clz: Type[T],
    file: Union[str, Path, None],
    overrides: Optional[List[str]] = None,
) -> T:
    if isinstance(file, str):
        file = Path(file)
    return _typed_load(clz, file=file, overrides=overrides)[0]


def find_latest_checkpoint(dir: Path) -> Optional[Path]:
    # TODO: error handling
    # Check that dir exists
    if not dir.exists():
        return None
    latest = None
    latest_dir = None
    for d in dir.iterdir():
        if d.is_dir() and len(d.name) >= 12:
            if latest is None or int(d.name[-12:]) > latest:
                latest = int(d.name[-12:])
                latest_dir = d
    return latest_dir


@dataclass
class OverridesDeserializer(Deserializer):
    overrides: List[str]
    applied_overrides: bool = False

    def deserialize(
        self,
        clz: Type[T],
        value: Any,
        path: str,
    ) -> Tuple[Optional[T], bool, bool]:
        if self.applied_overrides:
            return None, False, False
        for override in self.overrides:
            key, str_val = override.split("=", maxsplit=1)
            try:
                val = pyron.loads(str_val, preserve_structs=True, print_errors=False)
            except ValueError:
                val = str_val
            fpath = key.split(".")
            _value = value
            for segment in fpath[:-1]:
                if segment not in _value:
                    _value[segment] = {}
                _value = _value[segment]
            _value[fpath[-1]] = val
        self.applied_overrides = True
        return value, True, False


class ScheduleDeserializer(Deserializer):
    def __init__(self) -> None:
        # Use recursive type once supported (https://github.com/python/mypy/issues/731): ScheduleDict = Union[Dict[str, "ScheduleDict"], Schedule]
        self.schedules: Dict[str, Any] = {}

    def deserialize(
        self,
        clz: Type[T],
        value: Any,
        path: str,
    ) -> Tuple[Optional[T], bool, bool]:
        if (clz == int or clz == float) and isinstance(value, str) and "@" in value:
            schedule = _parse_schedule(value)
            field_name = path.split(".")[-1]

            def update(self: T, state: Any) -> None:
                x = getattr(state, schedule.xname)
                value = schedule.get_value(x)
                setattr(self, field_name, clz(value))  # type: ignore

            schedules = self.schedules
            for segment in path.split(".")[:-1]:
                if segment not in schedules:
                    schedules[segment] = {}
                schedules = self.schedules[segment]
            schedules[field_name] = Schedule(update, value)
            value = schedule.get_value(0.0)
            return clz(value), True, False  # type: ignore
        return None, False, False


@dataclass
class ScheduleSerializer(Serializer):
    # Use recursive type once supported (https://github.com/python/mypy/issues/731): ScheduleDict = Union[Dict[str, "ScheduleDict"], Schedule]
    schedules: Dict[str, Any]

    def serialize(self, value: Any, path: str, namedtuples: bool) -> Tuple[Any, bool]:
        segments = path.split(".")
        schedules = self.schedules
        for segment in segments:
            if segment not in schedules:
                return None, False
            schedules = schedules[segment]
        if isinstance(schedules, Schedule):
            return schedules.unparsed, True
        return None, False


def _dict_to_cpu(x: Any) -> Any:
    has_torch = False
    try:
        import torch  # pyright: reportMissingImports=false

        has_torch = True
    except ModuleNotFoundError:
        pass

    if has_torch and isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, dict):
        return {k: _dict_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_dict_to_cpu(v) for v in x]
    else:
        return x


@dataclass
class ElideDefaults(Serializer):
    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
    ) -> Tuple[Any, bool]:
        return None, False

    def modify_dataclass_attrs(
        self, value: Any, attrs: Dict[str, Any], path: str
    ) -> None:
        for name, field in value.__class__.__dataclass_fields__.items():
            if field.default is not MISSING and attrs[name] == field.default:
                del attrs[name]
            elif (
                field.default_factory is not MISSING
                and attrs[name] == field.default_factory()
            ):
                del attrs[name]
