from typing import Any, Sequence, TypeVar
from enum import EnumMeta
import enum
import typing
from abc import ABC
from dataclasses import dataclass, is_dataclass
import dataclasses
import docstring_parser

import pyron


T = TypeVar("T")


Type = typing.Union[
    "Primitive",
    "List",
    "Dict",
    "Union",
    "Struct",
    "Option",
    "Enum",
    "Literal",
    "Nothing",
]


@dataclass(eq=True, frozen=True)
class Primitive:
    type: str

    def __repr__(self) -> str:
        return self.type

    def is_subtype(self, other: Type) -> bool:
        return (
            (isinstance(other, Primitive) and other.type == self.type)
            or (isinstance(other, Option) and self.is_subtype(other.type))
            or (
                self.type == "int"
                and isinstance(other, Primitive)
                and other.type == "float"
            )
        )


@dataclass(eq=True, frozen=True)
class List:
    inner: Type

    def __repr__(self) -> str:
        return f"List[{self.inner}]"

    def is_subtype(self, other: Type) -> bool:
        return (
            self == other
            or (isinstance(other, List) and self.inner.is_subtype(other.inner))
            or (isinstance(other, Option) and self.is_subtype(other.type))
        )


@dataclass(eq=True, frozen=True)
class Dict:
    key: Type
    val: Type

    def __repr__(self) -> str:
        return f"Dict[{self.key}, {self.val}]"

    def is_subtype(self, other: Type) -> bool:
        return (
            self == other
            or (
                isinstance(other, Dict)
                and self.key == other.key
                and self.val.is_subtype(other.val)
            )
            or (isinstance(other, Option) and self.is_subtype(other.type))
        )


@dataclass(eq=True, frozen=True)
class Union:
    types: Sequence[Type]

    def __repr__(self) -> str:
        return f"Union[{', '.join(map(str, self.types))}]"

    def is_subtype(self, other: Type) -> bool:
        return (
            self == other
            or (
                isinstance(other, Union)
                and all(t.is_subtype(other) for t in self.types)
            )
            or (isinstance(other, Option) and self.is_subtype(other.type))
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
        ) or (isinstance(other, Option) and self.is_subtype(other.type))


@dataclass(frozen=True)
class Literal:
    allowed_values: typing.List[Any]

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
        ) or (isinstance(other, Option) and self.is_subtype(other.type))

    def __repr__(self) -> str:
        return "|".join(sorted([repr(v) for v in self.allowed_values]))

    def __eq__(self, other: Any) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return set(self.allowed_values) == set(other.allowed_values)


@dataclass(eq=True, frozen=True)
class Struct:
    name: str
    fields: typing.Dict[str, Field]
    version: typing.Optional[int] = 0

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.fields.items())})"

    def __str__(self) -> str:
        return self.name

    def find_field(self, path: Sequence[str]) -> typing.Optional[Field]:
        schema = self
        for segment in path[:-1]:
            if segment not in schema.fields:
                return None
            _schema = schema.fields[segment].type
            while isinstance(_schema, Option):
                _schema = _schema.type
            if isinstance(_schema, Struct):
                schema = _schema
            else:
                return None
        return schema.fields.get(path[-1])

    def is_subtype(self, other: Type) -> bool:
        return (
            isinstance(other, Struct)
            and all(
                k in other.fields
                and other.fields[k].type.is_subtype(self.fields[k].type)
                for k in self.fields
            )
            or (isinstance(other, Option) and self.is_subtype(other.type))
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

    def __repr__(self) -> str:
        return "Nothing"


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
    elif hasattr(clz, "__origin__") and clz.__origin__ == dict:
        return Dict(
            materialize_type(clz.__args__[0]), materialize_type(clz.__args__[1])
        )
    elif is_optional(clz):
        return Option(materialize_type(clz.__args__[0]))
    elif hasattr(clz, "__origin__") and clz.__origin__ == typing.Union:
        return Union([materialize_type(t) for t in clz.__args__])
    elif is_dataclass(clz):
        fields = {}
        field_docs = _find_all_field_docs(clz)
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
                field_docs.get(name),
            )
        from hyperstate.schema.versioned import Versioned

        return Struct(
            clz.__name__,
            fields,
            clz.version() if issubclass(clz, Versioned) else None,
        )
    elif isinstance(clz, EnumMeta):
        variants = {}
        for name, value in clz.__members__.items():
            variants[name] = value.value
        return Enum(clz.__name__, variants)
    elif typing.get_origin(clz) == typing.Literal:
        return Literal(list(typing.get_args(clz)))
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
        return Literal(list(schema.allowed_values))
    else:
        raise ValueError(f"Unsupported type: {clz_name}")


def load_schema(path: str) -> Struct:
    schema = pyron.load(path, preserve_structs=True)
    result = schema_from_namedtuple(schema)
    assert isinstance(result, Struct)
    return result


def _unwrap_container_type(type: Type) -> Type:
    if isinstance(type, List):
        return _unwrap_container_type(type.inner)
    if isinstance(type, Option):
        return _unwrap_container_type(type.type)
    return type


def _find_all_field_docs(clz: typing.Type[Any]) -> typing.Dict[str, str]:
    docs = docstring_parser.parse(clz.__doc__ or "")
    field_docs = {
        f.arg_name: f.description for f in docs.params if f.description is not None
    }
    if hasattr(clz, "__bases__"):
        for base in clz.__bases__:
            field_docs.update(_find_all_field_docs(base))
    return field_docs
