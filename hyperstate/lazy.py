from abc import ABC, abstractclassmethod, abstractmethod
import inspect
import os
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type,
    Generic,
    TypeVar,
)
from dataclasses import dataclass, field
from pathlib import Path
import msgpack

from hyperstate.serde import Serializer, Deserializer
from hyperstate import msgpack_torch


T = TypeVar("T")
C = TypeVar("C")
S = TypeVar("S")

# TODO: blob, lazy, and serializable should be orthogonal
class Serializable(ABC, Generic[C, S]):
    @abstractmethod
    def serialize(self) -> Any:
        pass

    @classmethod
    @abstractmethod
    def deserialize(
        cls: Type[T], state_dict: Any, config: C, state: S, ctx: Dict[str, Any]
    ) -> T:
        pass


class Lazy:
    def __getattribute__(self, name: str) -> Any:
        try:
            unloaded = super(Lazy, self).__getattribute__("_unloaded_lazy_fields")
        except AttributeError:
            unloaded = None
        if unloaded is not None and name in unloaded:
            ser_clz, config, path, legacy_pickle = unloaded[name]
            with open(path, "rb") as f:
                if legacy_pickle:
                    if "UNSAFE_DESERIALIZE_PICKLE" in os.environ:
                        import pickle

                        state_dict = pickle.load(f)
                    else:
                        raise RuntimeError(
                            "Deserialization of legacy snapshots containing pickled objects is not allowed as of hyperstate==0.4.0."
                            "To enable it, set the environment variable UNSAFE_DESERIALIZE_PICKLE to 1."
                        )
                else:
                    state_dict = msgpack.unpack(f, object_hook=msgpack_torch.decode)
            # TODO: recursion check
            value = ser_clz.deserialize(
                state_dict,
                config,
                self,
                self._deserialize_ctx if hasattr(self, "_deserialize_ctx") else {},
            )
            self.__setattr__(name, value)
            del unloaded[name]
        return super(Lazy, self).__getattribute__(name)

    def set_deserialize_ctx(self, key: str, value: Any) -> None:
        """
        Add a value to the deserialization context.

        :param key: The key to store the value under.
        :param value: The value to store.
        """
        if not hasattr(self, "_deserialize_ctx"):
            self._deserialize_ctx: Dict[str, Any] = {}
        self._deserialize_ctx[key] = value


@dataclass
class LazyDeserializer(Deserializer, Generic[C]):
    config: C
    path: Path
    lazy_fields: Dict[str, Tuple[Type[Serializable], C, Path, bool]] = field(
        default_factory=dict
    )

    def deserialize(
        self,
        cls: Type[T],
        value: Any,
        path: str,
    ) -> Tuple[Optional[T], bool, bool]:
        if inspect.isclass(cls) and issubclass(cls, Serializable):
            assert (
                value == "<BLOB>"
                or value == "<blob:pickle>"
                or value == "<blob:msgpack>"
            )
            if value == "<BLOB>":
                filepath = path.replace(".", "/").replace("[", "/").replace("]", "")
            elif value == "<blob:pickle>":
                filepath = (
                    "state." + path.replace("[", "_").replace("]", "") + ".pickle"
                )
            else:
                filepath = (
                    "state." + path.replace("[", "_").replace("]", "") + ".msgpack"
                )
            self.lazy_fields[path] = (
                cls,
                self.config,
                self.path / filepath,
                value == "<BLOB>" or value == "<blob:pickle>",
            )
            return None, True, True
        return None, False, False


@dataclass
class LazySerializer(Serializer):
    blobs: Dict[str, bytes] = field(default_factory=dict)

    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
    ) -> Tuple[Any, bool]:
        if isinstance(value, Serializable):
            path = "state." + path.replace("[", "_").replace("]", "") + ".msgpack"
            self.blobs[path] = msgpack.packb(
                value.serialize(), default=msgpack_torch.encode
            )
            return "<blob:msgpack>", True
        return None, False


def blob(clz: Type[T], mixin: Type[Serializable]) -> Type[T]:
    class Blob(mixin, clz):  # type: ignore
        pass

    return Blob
