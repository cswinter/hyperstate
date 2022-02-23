from typing import Any, TypeVar, Union
from enum import EnumMeta
import enum
import typing
from abc import ABC
from dataclasses import dataclass, is_dataclass
import dataclasses

import pyron


T = TypeVar("T")


Type = Union["Primitive", "List", "Struct", "Option", "Enum", "Literal", "Nothing"]


@dataclass(eq=True, frozen=True)
class Primitive:
    type: str

    def __repr__(self) -> str:
        return self.type

    def is_subtype(self, other: Type) -> bool:
        return (isinstance(other, Primitive) and other.type == self.type) or (
            isinstance(other, Option) and self.is_subtype(other.type)
        ) or (
            self.type == "int" and isinstance(other, Primitive) and other.type == "float"
        )


@dataclass(eq=True, frozen=True)
class List:
    inner: Type

    def __repr__(self) -> str:
        return f"List[{self.inner}]"

    def is_subtype(self, other: Type) -> bool:
        return self == other or (
            isinstance(other, Option) and self.is_subtype(other.type)
        )



@dataclass(eq=True, frozen=True)
class Field:
    name: str
    type: Type
    default: Any
    has_default: bool
    docstring: typing.Optional[str] = None


@dataclass(eq=True, frozen=True)
class Enum:
    name: str
    variants: typing.Dict[str, typing.Union[str, int]]

    def is_subtype(self, other: Type) -> bool:
        return (
            isinstance(other, Enum)
            and other.name == self.name
            and all(
                k in other.variants and other.variants[k] == v
                for k, v in self.variants.items()
            )
        ) or (
            isinstance(other, Option) and self.is_subtype(other.type)
        )



@dataclass(eq=True, frozen=True)
class Literal:
    allowed_values: typing.Set[Any]

    def is_subtype(self, other: Type) -> bool:
        return (
            (
                isinstance(other, Literal)
                and all(v in other.allowed_values for v in self.allowed_values)
            )
            or (
                isinstance(other, Primitive)
                and other.type == "str"
                and all(isinstance(v, str) for v in self.allowed_values)
            )
            or (
                isinstance(other, Primitive)
                and other.type == "int"
                and all(isinstance(v, int) for v in self.allowed_values)
            )
        ) or (
            isinstance(other, Option) and self.is_subtype(other.type)
        )



@dataclass(eq=True, frozen=True)
class Struct:
    name: str
    fields: typing.Dict[str, Field]
    version: typing.Optional[int] = 0

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.fields.items())})"

    def __str__(self) -> str:
        return self.name

    def is_subtype(self, other: Type) -> bool:
        return isinstance(other, Struct) and all(
            k in other.fields and other.fields[k].type.is_subtype(self.fields[k].type)
            for k in self.fields
        ) or (
            isinstance(other, Option) and self.is_subtype(other.type)
        )



@dataclass(eq=True, frozen=True)
class Option:
    type: Type

    def __repr__(self) -> str:
        return f"Optional[{self.type}]"

    def is_subtype(self, other: Type) -> bool:
        return isinstance(other, Option) and other.type.is_subtype(self.type)


@dataclass(eq=True, frozen=True)
class Nothing:
    def is_subtype(self, other: Type) -> bool:
        return True


def materialize_type(clz: typing.Type[Any]) -> Type:
    if clz == int:
        return Primitive(type="int")
    elif clz == str:
        return Primitive(type="str")
    elif clz == bool:
        return Primitive(type="bool")
    elif clz == float:
        return Primitive(type="float")
    elif hasattr(clz, "__origin__") and clz.__origin__ == list:
        return List(materialize_type(clz.__args__[0]))
    elif is_dataclass(clz):
        fields = {}
        docstrings = {}
        if clz.__doc__ is not None:
            for line in clz.__doc__.splitlines():
                line = line.strip()
                if line.startswith(":param"):
                    line = line[len(":param ") :].strip()
                    if ":" in line:
                        name, docs = line.split(":", 1)
                        name = name.strip()
                        docs = docs.strip()
                        docstrings[name] = docs
        for name, field in clz.__dataclass_fields__.items():
            if field.default is not dataclasses.MISSING:
                has_default = True
                default = field.default
            elif field.default_factory is not dataclasses.MISSING:
                has_default = True
                default = field.default_factory()
            else:
                has_default = False
                default = None
            if isinstance(default, enum.Enum):
                default = default.value
            fields[name] = Field(
                name,
                materialize_type(field.type),
                default,
                has_default,
                docstrings.get(name),
            )
        from hyperstate.schema.versioned import Versioned

        return Struct(
            clz.__name__,
            fields,
            clz.version() if issubclass(clz, Versioned) else None,
        )
    elif is_optional(clz):
        return Option(materialize_type(clz.__args__[0]))
    elif isinstance(clz, EnumMeta):
        variants = {}
        for name, value in clz.__members__.items():
            variants[name] = value.value
        return Enum(clz.__name__, variants)
    elif typing.get_origin(clz) == typing.Literal:
        return Literal(set(typing.get_args(clz)))
    else:
        raise ValueError(f"Unsupported type: {clz}")


def is_optional(clz: Any) -> bool:
    return (
        hasattr(clz, "__origin__")
        and clz.__origin__ is typing.Union
        and clz.__args__.__len__() == 2
        and clz.__args__[1] is type(None)
    )


def schema_from_namedtuple(schema: Any) -> Type:
    clz_name = schema.__class__.__name__
    if clz_name == "Primitive":
        return Primitive(schema.type)
    elif clz_name == "List":
        return List(schema_from_namedtuple(schema.inner))
    elif clz_name == "Struct":
        fields = {}
        for name, field in schema.fields.items():
            fields[name] = Field(
                name,
                schema_from_namedtuple(field.type),
                field.default,
                field.has_default,
            )
        return Struct(schema.name, fields, schema.version)
    elif clz_name == "Option":
        return Option(schema_from_namedtuple(schema.type))
    elif clz_name == "Enum":
        variants = {}
        for name, value in schema.variants.items():
            variants[name] = value
        return Enum(schema.name, variants)
    elif clz_name == "Literal":
        return Literal(set(schema.allowed_values))
    else:
        raise ValueError(f"Unsupported type: {clz_name}")


def load_schema(path: str) -> Type:
    with open(path, "r") as f:
        schema = pyron.load(f.read(), preserve_structs=True)
    return schema_from_namedtuple(schema)
