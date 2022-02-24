from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum, EnumMeta
from os import name
from pathlib import Path
from typing import (
    Iterable,
    List,
    Any,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    Dict,
    Tuple,
    Union,
)
import inspect
from dataclasses import MISSING, is_dataclass
import typing

import pyron

T = TypeVar("T")


class Serializer(ABC):
    @abstractmethod
    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
    ) -> Tuple[Any, bool]:
        pass

    def modify_dataclass_attrs(
        self, value: Any, attrs: Dict[str, Any], path: str
    ) -> None:
        pass


class Deserializer(ABC):
    @abstractmethod
    def deserialize(
        self,
        clz: Type[T],
        value: Any,
        path: str,
    ) -> Tuple[Optional[T], bool, bool]:
        pass


def asdict(
    value: Any,
    named_tuples: bool = False,
    serializers: Optional[List[Serializer]] = None,
    file: str = "",
) -> Any:
    if serializers is None:
        serializers = []

    for serializer in serializers:
        _value, _ok = serializer.serialize(value, file, named_tuples)
        if _ok:
            value = _value
    if is_dataclass(value):
        attrs = {
            field_name: asdict(
                value=getattr(value, field_name),
                named_tuples=named_tuples,
                serializers=serializers,
                file=field_name if file == "" else f"{file}.{field_name}",
            )
            for field_name in value.__dataclass_fields__
        }
        for serializer in serializers:
            serializer.modify_dataclass_attrs(value, attrs, file)
        if named_tuples:
            return namedtuple(value.__class__.__name__, attrs.keys())(**attrs)
        else:
            return attrs
    elif isinstance(value, dict) or isinstance(value, list):
        # TODO: recurse
        return value
    elif isinstance(value, Enum):
        return value.name
    elif (
        isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, str)
        or isinstance(value, bool)
        or value is None
    ):
        return value
    else:
        raise TypeError(f"Can't serialize value {value} of type {type(value)}")


def from_dict(
    clz: Type[T],
    value: Any,
    deserializers: Optional[List[Deserializer]] = None,
    fpath: str = "",
    ignore_extra_fields: bool = False,
) -> T:
    if deserializers is None:
        deserializers = []

    if is_optional(clz):
        if value is None:
            return None  # type: ignore
        else:
            clz = clz.__args__[0]  # type: ignore

    ret = False
    for deserializer in deserializers:
        _value, ok, _ret = deserializer.deserialize(clz, value, fpath)
        if ok:
            value = _value
        ret = ret or _ret
    if ret:
        return _value  # type: ignore
    if inspect.isclass(clz) and isinstance(value, clz):
        return value
    elif clz == str and isnamedtupleinstance(value) and len(value._fields) == 0:
        return value.__class__.__name__  # type: ignore
    elif clz == float and isinstance(value, int):
        return float(value)  # type: ignore
    elif clz == int and isinstance(value, float) and int(value) == value:
        return int(value)  # type: ignore
    elif clz == float and isinstance(value, str):
        return float(value)  # type: ignore
    elif clz == int and isinstance(value, str):
        f = float(value)
        if int(f) == f:
            return int(f)  # type: ignore
        else:
            raise ValueError(f"Expected {fpath} to be an int, got {value}")
    elif (
        hasattr(clz, "__args__")
        and len(clz.__args__) == 1  # type: ignore
        and clz == List[clz.__args__]  # type: ignore
        and isinstance(value, list)
    ):
        # TODO: recurse
        return value  # type: ignore
    elif (
        hasattr(clz, "__args__")
        and len(clz.__args__) == 2  # type: ignore
        and clz == Dict[clz.__args__]  # type: ignore
        and isinstance(value, dict)
    ):
        # TODO: recurse
        return value  # type: ignore
    elif is_dataclass(clz):
        # TODO: better error
        if value == ():
            value = {}
        elif isnamedtupleinstance(value):
            value = value._asdict()
        if not isinstance(value, dict):
            raise ValueError(f"'{value}' cannot be deserialized as '{clz.__name__}'.")
        kwargs = {}
        remaining_fields = set(clz.__dataclass_fields__.keys())  # type: ignore
        for field_name, v in value.items():
            field = clz.__dataclass_fields__.get(field_name)  # type: ignore
            if field is None:
                if ignore_extra_fields:
                    continue
                else:
                    raise TypeError(
                        f"{clz.__module__}.{clz.__name__} has no attribute {field_name}."
                    )
            remaining_fields.remove(field_name)
            kwargs[field_name] = from_dict(
                clz=field.type,
                value=v,
                deserializers=deserializers,
                fpath=f"{fpath}.{field_name}" if fpath else field_name,
            )
        _try_create_defaults(clz, kwargs, remaining_fields, fpath)
        try:
            instance = clz(**kwargs)  # type: ignore
            return instance
        except TypeError as e:
            raise TypeError(f"Failed to initialize '{clz.__name__}': {e}")
    elif isinstance(clz, EnumMeta):
        return clz(value)
    elif typing.get_origin(clz) == Literal:
        args = typing.get_args(clz)
        if value not in args:
            raise ValueError(f"{value} must be one of {args}")
        return value  # type: ignore
    if isnamedtupleinstance(value) and len(value) == 0:
        value = value.__class__.__name__
    raise TypeError(
        f"Failed to deserialize '{fpath}': {repr(value)} is not a {_qualified_name(clz)}"
    )


def _try_create_defaults(
    cls: Any,
    kwargs: Dict[str, Any],
    remaining_fields: Iterable[Any],
    fpath: str,
) -> Any:
    still_missing = []
    for _field_name in remaining_fields:
        field = cls.__dataclass_fields__.get(_field_name)
        if field.default is MISSING and field.default_factory is MISSING:
            if is_dataclass(field.type):
                _missing_defaults = [
                    f.name
                    for f in field.type.__dataclass_fields__.values()
                    if f.default is MISSING and f.default_factory is MISSING
                ]
                _kwargs: Dict[str, Any] = {}
                _try_create_defaults(
                    field.type,
                    _kwargs,
                    _missing_defaults,
                    f"{fpath}.{_field_name}" if fpath else _field_name,
                )
                kwargs[field.name] = field.type(**_kwargs)
            else:
                still_missing.append(_field_name)

    if len(still_missing) > 0:
        prefix = f"{fpath}." if fpath else ""
        missing_fields = ", ".join([f"'{prefix}{f}'" for f in still_missing])
        raise TypeError(
            f"Failed to initialize '{cls.__name__}': missing value for {missing_fields}"
        )


def loads(
    clz: Type[T],
    data: str,
    deserializers: Optional[List[Deserializer]] = None,
    ignore_extra_fields: bool = False,
) -> T:
    if deserializers is None:
        deserializers = []
    return from_dict(
        clz, pyron.loads(data), deserializers, ignore_extra_fields=ignore_extra_fields
    )


def load(
    clz: Type[T],
    file: Union[str, Path],
    deserializers: Optional[List[Deserializer]] = None,
    ignore_extra_fields: bool = False,
) -> T:
    if deserializers is None:
        deserializers = []
    return from_dict(
        clz,
        pyron.load(str(file)),
        deserializers,
        ignore_extra_fields=ignore_extra_fields,
    )


def dump(
    obj: Any,
    file: Union[str, Path],
    serializers: Optional[List[Serializer]] = None,
) -> str:
    serialized = dumps(obj, serializers)
    with open(file, "w") as f:
        f.write(serialized)
    return serialized


def dumps(
    obj: Any,
    serializers: Optional[List[Serializer]] = None,
) -> str:
    if serializers is None:
        serializers = []
    state_dict = asdict(obj, named_tuples=True, serializers=serializers)
    return pyron.to_string(state_dict)


def is_optional(clz: Type[Any]) -> bool:
    return (
        hasattr(clz, "__origin__")
        and clz.__origin__ is Union
        and clz.__args__.__len__() == 2
        and clz.__args__[1] is type(None)
    )


def _qualified_name(clz: Type[Any]) -> str:
    if clz.__module__ == "builtin":
        return clz.__name__
    elif not hasattr(clz, "__module__") or not hasattr(clz, "__name__"):
        return repr(clz)
    else:
        return f"{clz.__module__}.{clz.__name__}"


def isnamedtupleinstance(x: Any) -> bool:
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)
