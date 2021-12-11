from typing import Sequence, Type, Any, cast
from enum import Enum
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import types as t
from .rewrite_rule import (
    RewriteRule,
    DeleteField,
    RenameField,
    MapFieldValue,
    ChangeDefault,
    AddDefault,
)

import click


class Severity(Enum):
    INFO = 1
    WARN = 2
    ERROR = 3

    def __ge__(self, other: "Severity") -> bool:
        return cast(bool, self.value >= other.value)

    def __gt__(self, other: "Severity") -> bool:
        return cast(bool, self.value > other.value)

    def __le__(self, other: "Severity") -> bool:
        return cast(bool, self.value <= other.value)

    def __lt__(self, other: "Severity") -> bool:
        return cast(bool, self.value < other.value)


# https://github.com/python/mypy/issues/5374
@dataclass(eq=True, frozen=True)  # type: ignore
class SchemaChange(ABC):
    field: Sequence[str]

    @abstractmethod
    def diagnostic(self) -> str:
        pass

    def severity(self) -> Severity:
        if self.proposed_fix() is not None:
            return Severity.WARN
        else:
            return Severity.ERROR

    def proposed_fix(self) -> typing.Optional[RewriteRule]:
        return None

    @property
    def field_name(self) -> str:
        return ".".join(self.field)

    def emit_diagnostic(self) -> None:
        severity = self.severity()
        if severity == Severity.INFO:
            styled_severity = click.style("INFO ", fg="white")
        elif severity == Severity.WARN:
            styled_severity = click.style("WARN ", fg="yellow")
        else:
            styled_severity = click.style("ERROR", fg="red")

        print(
            f"{styled_severity} {self.diagnostic()}: {click.style(self.field_name, fg='cyan')}"
        )


@dataclass(eq=True, frozen=True)
class FieldAdded(SchemaChange):
    type: t.Type
    default: Any = None
    has_default: bool = False

    def severity(self) -> Severity:
        if self.has_default:
            return Severity.INFO
        return super().severity()

    def proposed_fix(self) -> typing.Optional[AddDefault]:
        if self.has_default:
            return None
        elif isinstance(self.type, t.Option):
            return AddDefault(self.field, None)
        elif isinstance(self.type, t.List):
            return AddDefault(self.field, [])
        return None

    def diagnostic(self) -> str:
        return "new field without default value"


@dataclass(eq=True, frozen=True)
class FieldRemoved(SchemaChange):
    type: t.Type
    default: Any = None
    has_default: bool = False

    def diagnostic(self) -> str:
        return "field removed"

    def proposed_fix(self) -> RewriteRule:
        return DeleteField(self.field)


@dataclass(eq=True, frozen=True)
class FieldRenamed(SchemaChange):
    new_name: Sequence[str]

    def diagnostic(self) -> str:
        return f"field renamed to {'.'.join(self.new_name)}"

    def proposed_fix(self) -> RewriteRule:
        return RenameField(self.field, self.new_name)


@dataclass(eq=True, frozen=True)
class DefaultValueChanged(SchemaChange):
    old: Any
    new: Any

    def diagnostic(self) -> str:
        return f"default value changed from {self.old} to {self.new}"

    def proposed_fix(self) -> RewriteRule:
        return ChangeDefault(self.field, self.new)


@dataclass(eq=True, frozen=True)
class DefaultValueRemoved(SchemaChange):
    old: Any

    def diagnostic(self) -> str:
        return f"default value removed: {self.old}"

    def proposed_fix(self) -> RewriteRule:
        return AddDefault(self.field, self.old)


@dataclass(eq=True, frozen=True)
class TypeChanged(SchemaChange):
    old: t.Type
    new: t.Type

    def severity(self) -> Severity:
        if isinstance(self.new, t.Option) and self.new.type == self.old:
            return Severity.INFO
        elif (
            isinstance(self.old, t.Primitive)
            and isinstance(self.new, t.Primitive)
            and self.old.type == "int"
            and self.new.type == "float"
        ):
            return Severity.INFO
        return super().severity()

    def diagnostic(self) -> str:
        return f"type changed from {self.old} to {self.new}"

    def proposed_fix(self) -> typing.Optional[RewriteRule]:
        if isinstance(self.new, t.List) and self.new.inner == self.old:
            return MapFieldValue(self.field, lambda x: [x], rendered="lambda x: [x]")
        elif (
            isinstance(self.old, t.Primitive)
            and isinstance(self.new, t.Primitive)
            and self.old == "float"
            and self.new == "int"
        ):
            return MapFieldValue(
                self.field, lambda x: int(x), rendered="lambda x: int(x)"
            )
        else:
            return None


@dataclass(eq=True, frozen=True)
class EnumVariantValueChanged(SchemaChange):
    enum_name: str
    variant: str
    old_value: typing.Union[str, int]
    new_value: typing.Union[str, int]

    def diagnostic(self) -> str:
        return f"value of {self.enum_name}.{self.variant} changed from {self.old_value} to {self.new_value}"

    def proposed_fix(self) -> RewriteRule:
        return MapFieldValue(
            self.field,
            lambda x: x if x != self.old_value else self.new_value,
            rendered=f"lambda x: x if x != {self.old_value.__repr__()} else {self.new_value.__repr__()}",
        )


@dataclass(eq=True, frozen=True)
class EnumVariantRemoved(SchemaChange):
    enum_name: str
    variant: str
    variant_value: typing.Union[str, int]

    def diagnostic(self) -> str:
        return f"variant {self.variant} of {self.enum_name} removed"


@dataclass(eq=True, frozen=True)
class EnumVariantAdded(SchemaChange):
    enum_name: str
    variant: str
    variant_value: typing.Union[str, int]

    def severity(self) -> Severity:
        return Severity.INFO

    def diagnostic(self) -> str:
        return f"variant {self.variant} of {self.enum_name} added"


@dataclass(eq=True, frozen=True)
class EnumVariantRenamed(SchemaChange):
    enum_name: str
    old_variant_name: str
    new_variant_name: str

    def severity(self) -> Severity:
        return Severity.INFO

    def diagnostic(self) -> str:
        return f"variant {self.new_variant_name} of {self.enum_name} renamed to {self.new_variant_name}"
