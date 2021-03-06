from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from inspect import isclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Tuple
from hyperstate.schema import namedtuple_utils, types
from hyperstate.serde import (
    Deserializer,
    Serializer,
)

from hyperstate.schema.rewrite_rule import RewriteRule

T = TypeVar("T")


@dataclass  # type: ignore
class Versioned(ABC):
    @classmethod
    @abstractmethod
    def version(clz) -> int:
        raise NotImplementedError(f"{clz.__name__}.version() not implemented")

    @classmethod
    def minimum_version(clz) -> int:
        return 0

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        """
        Returns a list of rewrite rules that can be applied to the given version
        to make it compatible with the next version.
        """
        return {}

    @classmethod
    def _apply_upgrades(clz, state_dict: Any, version: int) -> Any:
        for v in clz.upgrade_rules().keys():
            assert (
                v < clz.version()
            ), f"{clz.__name__}.upgrade_rules() keys must be less than {clz.__name__}.version()"
        for i in range(version, clz.version()):
            for rule in clz.upgrade_rules().get(i, []):
                state_dict = rule.apply(state_dict)
        return state_dict

    @classmethod
    def _apply_schema_upgrades(clz, schema: types.Struct) -> None:
        for i in range(schema.version or 0, clz.version()):
            for rule in clz.upgrade_rules().get(i, []):
                rule.apply_to_schema(schema)


@dataclass
class VersionedDeserializer(Deserializer):
    allow_missing_version: bool = False

    def deserialize(
        self,
        clz: Type[T],
        value: Any,
        path: str,
    ) -> Tuple[Optional[T], bool, bool]:
        if isclass(clz) and issubclass(clz, Versioned):
            if isinstance(value, dict):
                version = value.pop("version", None)
            elif hasattr(value, "version"):
                version = value.version
                value = namedtuple_utils.remove_field(value, "version")
            else:
                version = None

            if version is None:
                if not self.allow_missing_version:
                    raise ValueError(
                        f"Value for versioned class {clz.__name__} missing `version` field."
                    )
            else:
                value = clz._apply_upgrades(state_dict=value, version=version)
        return None, False, False


@dataclass
class VersionedSerializer(Serializer):
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
        if isinstance(value, Versioned):
            # Insert version as first element so it shows up at start of config file
            _attrs = dict(attrs)
            attrs.clear()
            attrs["version"] = value.__class__.version()
            for k, v in _attrs.items():
                attrs[k] = v
