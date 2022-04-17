import os
import errno
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
from typing import (
    Callable,
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
import uuid
from hyperstate.schema.types import Struct, materialize_type
import hyperstate.schema.types as t

from hyperstate.schema.versioned import (
    Versioned,
    VersionedSerializer,
    VersionedDeserializer,
)
from hyperstate.serde import (
    Deserializer,
    Serializer,
    asdict,
)
import hyperstate.serde as serde
from .lazy import Lazy, LazyDeserializer, LazySerializer
import pyron

from hyperstate.schedule import Schedule, _parse_schedule

C = TypeVar("C")
S = TypeVar("S")
T = TypeVar("T")


class StateManager(Generic[C, S]):
    def __init__(
        self,
        config_cls: Type[C],
        state_cls: Type[S],
        initial_state: Callable[[C, Dict[str, Any]], S],
        init_path: Union[str, Path, None],
        checkpoint_dir: Union[str, Path, None] = None,
        overrides: Optional[List[str]] = None,
        ignore_extra_fields: bool = False,
        checkpoint_key: str = "step",
    ) -> None:
        """
        :param config_clz: The type of the config object.
        :param state_clz: The type of the state object.
        :param initial_state: A function that takes the config object and returns the initial state.
        :param init_path: Path to a config file or checkpoint.
        :param checkpoint_dir: Directory to store checkpoints. If the directory contains a valid checkpoint, the latest checkpoint will be loaded and `init_path` will be ignored.
        :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
        :param ignore_extra_fields: If `True`, ignore extra fields in the config file.
        :param checkpoint_key: The name of a field in the state object that will have a unique value for each checkpoint.
        """
        self.config_cls = config_cls
        self.state_cls = state_cls
        self._initial_state = initial_state
        self.checkpoint_key = checkpoint_key

        self.init_path = Path(init_path) if init_path is not None else None
        self._last_checkpoint: Optional[Path] = None
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.overrides = overrides
        self.ignore_extra_fields = ignore_extra_fields

        self._config: Optional[C] = None
        self._schedules: Dict[str, Any] = {}
        self._state: Optional[S] = None
        self._deserialize_ctx: Dict[str, Any] = {}

    @property
    def config(self) -> C:
        if self._config is None:
            self._config, self._schedules = self._load_config()
        return self._config

    @property
    def state(self) -> S:
        if self._state is None:
            self._state = self._load_state()
            if isinstance(self._state, Lazy):
                self._state._deserialize_ctx = {**self._deserialize_ctx}
            _apply_schedules(self.state, self.config, self._schedules)
        return self._state  # type: ignore

    @property
    def checkpoint_dir(self) -> Union[Path, None]:
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, value: Union[Path, None]) -> None:
        if value is not None:
            value.mkdir(parents=True, exist_ok=True)
            checkpoint = find_latest_checkpoint(value)
            if checkpoint is not None and checkpoint.name.startswith("latest"):
                self._last_checkpoint = checkpoint
        else:
            self._checkpoint_dir = None
        self._checkpoint_dir = value

    def set_deserialize_ctx(self, key: str, value: Any) -> None:
        """
        Add a value to the deserialization context.

        :param key: The key to store the value under.
        :param value: The value to store.
        """
        self._deserialize_ctx[key] = value

    def _load_config(self) -> Tuple[C, Dict[str, Any]]:
        if self._last_checkpoint is None:
            path = self.init_path
        else:
            path = self._last_checkpoint
        if path is not None and path.is_dir():
            path = path / "config.ron"
        return _typed_load(
            self.config_cls,
            file=path,
            overrides=self.overrides or [],
            allow_missing_version=path is None,
            ignore_extra_fields=self.ignore_extra_fields,
        )

    def _load_state(self) -> S:
        if self._last_checkpoint is not None:
            path = self._last_checkpoint
        elif self.init_path is not None and self.init_path.is_dir():
            path = self.init_path
        else:
            return self._initial_state(self.config, self._deserialize_ctx)
        return _typed_load(
            self.state_cls,
            file=path / "state.ron",
            config=self.config,
            ignore_extra_fields=self.ignore_extra_fields,
        )[0]

    def step(self) -> None:
        """
        Updates all hyperparameter schedules and persists the current state.
        """
        _apply_schedules(self.state, self.config, self._schedules)
        if self.checkpoint_dir is not None:
            val = getattr(self.state, self.checkpoint_key)
            assert isinstance(
                val, int
            ), f"checkpoint key `{self.checkpoint_key}` must be an integer, but found value `{val}` of type `{type(val)}`"
            checkpoint_dir = (
                self.checkpoint_dir / f"latest-{self.checkpoint_key}{val:012}"
            )
            if not checkpoint_dir.exists():
                self.checkpoint(str(checkpoint_dir))
            if self._last_checkpoint is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.move(str(self._last_checkpoint), tmpdir)
            self._last_checkpoint = checkpoint_dir
            # TODO: persistent checkpoints

    def checkpoint(self, target_dir: str) -> None:
        """
        Persist the current state to a checkpoint.

        :param target_dir: The directory to store the checkpoint.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "checkpoint"
            p.mkdir()
            _typed_dump(self.config, p / "config.ron", self._schedules)
            _typed_dump(self.state, p / "state.ron")
            atomic_move(str(p), target_dir)

    def config_dict(self) -> Any:
        return asdict(self.config, serializers=[VersionedSerializer()])


class FieldNotFoundError(Exception):
    def __init__(self, fpath: str, cls: Type[Any], fname: str) -> None:
        super().__init__(f"Unknown field '{fpath}' for class '{cls.__name__}")
        self.field_name = fname
        self.fpath = fpath
        self.cls = cls


class FieldsNotFoundError(Exception):
    def __init__(self, not_found_errors: List[FieldNotFoundError]) -> None:
        super().__init__(f"Unknown fields: {[e.fpath for e in not_found_errors]}")
        self.not_found_errors = not_found_errors


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
    if file is None and issubclass(clz, Versioned):
        allow_missing_version = True
    else:
        allow_missing_version = False
    if isinstance(file, str):
        file = Path(file)
    return _typed_load(
        clz, file=file, overrides=overrides, allow_missing_version=allow_missing_version
    )[0]


def find_latest_checkpoint(dir: Path) -> Optional[Path]:
    # Check that dir exists
    if not dir.exists():
        return None
    latest = None
    latest_dir = None
    for d in dir.iterdir():
        if d.is_dir() and len(d.name) >= 12:
            try:
                if latest is None or int(d.name[-12:]) > latest:
                    latest = int(d.name[-12:])
                    latest_dir = d
            except ValueError:
                pass
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

        schema = materialize_type(clz)
        errors = []
        assert isinstance(schema, Struct), f"{clz} is not a struct"
        for override in self.overrides:
            keyval = override.split("=", maxsplit=1)
            if len(keyval) == 1:
                raise ValueError(
                    f"Invalid override: {override}. Expected format: field.name=value"
                )
            key, str_val = keyval
            fpath = key.split(".")
            field = schema.find_field(fpath)

            if field is None:
                errors.append(FieldNotFoundError(key, clz, fpath[-1]))
                continue

            if isinstance(field.type, t.Primitive) and field.type.type == "str":
                val: Any = str_val
            elif isinstance(field.type, t.Primitive) and field.type.type == "bool":
                if (
                    str_val == "True"
                    or str_val == "true"
                    or str_val == "1"
                    or str_val == "t"
                    or str_val == "T"
                ):
                    val = True
                elif (
                    str_val == "False"
                    or str_val == "false"
                    or str_val == "0"
                    or str_val == "f"
                    or str_val == "F"
                ):
                    val = False
            else:
                val = pyron.loads(str_val, preserve_structs=True, print_errors=False)
            _value = value
            for segment in fpath[:-1]:
                if segment not in _value:
                    _value[segment] = {}
                _value = _value[segment]
            _value[fpath[-1]] = val
        if len(errors) == 1:
            raise errors[0]
        elif len(errors) > 1:
            raise FieldsNotFoundError(errors)
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


def atomic_move(src: str, dst: str) -> None:
    """Rename a file from ``src`` to ``dst``.

    *   Moves must be atomic.  ``shutil.move()`` is not atomic.
        Note that multiple threads may try to write to the cache at once,
        so atomicity is required to ensure the serving on one thread doesn't
        pick up a partially saved image from another thread.

    *   Moves must work across filesystems.  Often temp directories and the
        cache directories live on different filesystems.  ``os.rename()`` can
        throw errors if run across filesystems.

    So we try ``os.rename()``, but if we detect a cross-filesystem copy, we
    switch to ``shutil.move()`` with some wrappers to make it atomic.
    """
    try:
        os.rename(src, dst)
    except OSError as err:

        if err.errno == errno.EXDEV:
            # Generate a unique ID, and copy `<src>` to the target directory
            # with a temporary name `<dst>.<ID>.tmp`.  Because we're copying
            # across a filesystem boundary, this initial copy may not be
            # atomic.  We intersperse a random UUID so if different processes
            # are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = f"{dst}.{copy_id}.tmp"
            shutil.move(src, tmp_dst)

            # Then do an atomic rename onto the new name, and clean up the
            # source image.
            os.rename(tmp_dst, dst)
        else:
            raise
