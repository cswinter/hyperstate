from typing import Callable, List, Optional, Tuple, Any, Type
import typing
from dataclasses import is_dataclass
from pathlib import Path
import difflib

import pyron
import click

from hyperstate.hyperstate import _typed_dump, _typed_load
from hyperstate.schema.schema_change import (
    DefaultValueChanged,
    DefaultValueRemoved,
    EnumVariantAdded,
    EnumVariantRemoved,
    EnumVariantRenamed,
    EnumVariantValueChanged,
    FieldAdded,
    FieldRemoved,
    FieldRenamed,
    SchemaChange,
    Severity,
    TypeChanged,
)
from hyperstate.schema.versioned import Versioned
from .types import load_schema, materialize_type
from . import types as t

TAdded = typing.TypeVar("TAdded", bound="SchemaChange")
TRemoved = typing.TypeVar("TRemoved", bound="SchemaChange")


class SchemaChecker:
    def __init__(
        self,
        old: t.Struct,
        config_clz: Type[Versioned],
        perform_upgrade: bool = True,
    ):
        self.config_clz = config_clz
        new = materialize_type(config_clz)
        assert isinstance(new, t.Struct)
        self.new = new
        if perform_upgrade:
            config_clz._apply_schema_upgrades(old)
        self.old = old
        self.changes: List[SchemaChange] = []
        self.proposed_fixes = []
        self._find_changes(old, self.new, [])
        self._find_field_renames()
        self._find_enum_variant_renames()
        for change in self.changes:
            proposed_fix = change.proposed_fix()
            if proposed_fix is not None:
                self.proposed_fixes.append(proposed_fix)

    def severity(self) -> Severity:
        max_severity = Severity.INFO
        for change in self.changes:
            if change.severity() > max_severity:
                max_severity = change.severity()
        return max_severity

    def print_report(self) -> None:
        for change in self.changes:
            change.emit_diagnostic()
        if self.severity() > Severity.INFO and self.old.version == self.new.version:
            print(
                click.style("WARN", fg="yellow")
                + "  schema changed but version identical"
            )

        if self.severity() == Severity.INFO:
            click.secho("Schema compatible", fg="green")
        else:
            click.secho("Schema incompatible", fg="red")
            print()
            click.secho("Proposed mitigations", fg="white", bold=True)
            if self.proposed_fixes:
                click.secho("- add upgrade rules:", fg="white", bold=True)
                print(f"    {self.old.version}: [")
                for mitigation in self.proposed_fixes:
                    print(f"        {mitigation},")
                print("    ],")
            if self.severity() > Severity.INFO and self.old.version == self.new.version:
                click.secho(
                    f"- bump version to {self.old.version or -1 + 1}",
                    fg="white",
                    bold=True,
                )

    def _find_changes(self, old: t.Type, new: t.Type, path: List[str]) -> None:
        if old.__class__ != new.__class__:
            self.changes.append(TypeChanged(tuple(path), old, new))
        elif isinstance(old, t.Primitive):
            if old != new:
                self.changes.append(TypeChanged(tuple(path), old, new))
        elif isinstance(old, t.List):
            assert isinstance(new, t.List)
            self._find_changes(old.inner, new.inner, path + ["[]"])
        elif isinstance(old, t.Struct):
            assert isinstance(new, t.Struct)
            for name, field in new.fields.items():
                if name not in old.fields:
                    if isinstance(field.type, t.Struct):
                        self._all_new(field.type, path + [name])
                    else:
                        self.changes.append(
                            FieldAdded(
                                tuple(path + [name]),
                                field.type,
                                field.default,
                                field.has_default,
                            )
                        )
                else:
                    oldfield = old.fields[name]
                    if isinstance(field.type, t.Struct) and not isinstance(
                        oldfield.type, t.Struct
                    ):
                        self.changes.append(
                            FieldRemoved(
                                tuple(path + [name]),
                                oldfield.type,
                                oldfield.default,
                                oldfield.has_default,
                            )
                        )
                        self._all_new(field.type, path + [name])
                    elif isinstance(oldfield.type, t.Struct) and not isinstance(
                        field.type, t.Struct
                    ):
                        self.changes.append(
                            FieldAdded(
                                tuple(path + [name]),
                                field.type,
                                field.default,
                                field.has_default,
                            )
                        )
                        self._all_gone(oldfield.type, path + [name])
                    else:
                        self._find_changes(oldfield.type, field.type, path + [name])
                        if oldfield.has_default != field.has_default:
                            if not field.has_default:
                                self.changes.append(
                                    DefaultValueRemoved(
                                        tuple(path + [name]), oldfield.default
                                    )
                                )
                        elif oldfield.has_default and oldfield.default != field.default:
                            if is_dataclass(field.default):
                                # TODO: perform comparison against namedtuple
                                pass
                            else:
                                self.changes.append(
                                    DefaultValueChanged(
                                        tuple(path + [name]),
                                        oldfield.default,
                                        field.default,
                                    )
                                )
            # Check for removed fields
            for name, field in old.fields.items():
                if name not in new.fields:
                    if isinstance(field.type, t.Struct):
                        self._all_gone(field.type, path + [name])
                    else:
                        self.changes.append(
                            FieldRemoved(
                                tuple(
                                    path + [name],
                                ),
                                field.type,
                                field.default,
                                field.has_default,
                            )
                        )

        elif isinstance(old, t.Option):
            assert isinstance(new, t.Option)
            self._find_changes(old.type, new.type, path + ["?"])
        elif isinstance(old, t.Enum):
            assert isinstance(new, t.Enum)
            for name, value in new.variants.items():
                if name not in old.variants:
                    self.changes.append(
                        EnumVariantAdded(
                            tuple(path),
                            new.name,
                            name,
                            value,
                        )
                    )
                else:
                    if old.variants[name] != value:
                        self.changes.append(
                            EnumVariantValueChanged(
                                tuple(path),
                                new.name,
                                name,
                                old.variants[name],
                                value,
                            )
                        )
            for name, value in old.variants.items():
                if name not in new.variants:
                    self.changes.append(
                        EnumVariantRemoved(tuple(path), new.name, name, value)
                    )
        else:
            raise ValueError(f"Field {'.'.join(path)} has unsupported type {type(old)}")

    def _find_renames(
        self,
        cls_added: Type[TAdded],
        cls_removed: Type[TRemoved],
        similarity: Callable[[TAdded, TRemoved], Optional[float]],
        new_renamed: Callable[[TAdded, TRemoved], SchemaChange],
    ) -> None:
        threshold = 0.1
        removeds: List[TRemoved] = [
            change for change in self.changes if isinstance(change, cls_removed)
        ]
        addeds: List[TAdded] = [
            change for change in self.changes if isinstance(change, cls_added)
        ]
        # TODO: O(n^3). can be implemented in O(n^2 log n)
        while True:
            best_similarity = threshold
            best_match: Optional[Tuple[TRemoved, TAdded]] = None
            for removed in removeds:
                for added in addeds:
                    simi = similarity(added, removed)
                    if simi is not None and simi > best_similarity:
                        best_similarity = simi
                        best_match = (removed, added)

            if best_match is None:
                break
            removed, added = best_match
            self.changes.remove(removed)
            self.changes.remove(added)
            removeds.remove(removed)
            addeds.remove(added)
            self.changes.append(new_renamed(added, removed))

    def _find_field_renames(self) -> None:
        def field_similarity(
            added: FieldAdded, removed: FieldRemoved
        ) -> Optional[float]:
            if (
                removed.type == added.type
                and removed.has_default == added.has_default
                and removed.default == added.default
            ):
                return name_similarity(removed.field[-1], added.field[-1])
            return None

        def field_renamed(added: FieldAdded, removed: FieldRemoved) -> SchemaChange:
            return FieldRenamed(removed.field, added.field)

        self._find_renames(FieldAdded, FieldRemoved, field_similarity, field_renamed)

    def _find_enum_variant_renames(self) -> None:
        def enum_similarity(
            added: EnumVariantAdded, removed: EnumVariantRemoved
        ) -> Optional[float]:
            return name_similarity(removed.variant, added.variant)

        def enum_renamed(
            added: EnumVariantAdded, removed: EnumVariantRemoved
        ) -> SchemaChange:
            if removed.variant_value != added.variant_value:
                return EnumVariantValueChanged(
                    removed.field,
                    added.enum_name,
                    removed.variant,
                    removed.variant_value,
                    added.variant_value,
                )
            else:
                return EnumVariantRenamed(
                    added.field, added.enum_name, removed.variant, added.variant
                )

        self._find_renames(
            EnumVariantAdded, EnumVariantRemoved, enum_similarity, enum_renamed
        )

    def _all_new(self, struct: t.Struct, path: typing.List[str]) -> None:
        for name, field in struct.fields.items():
            if isinstance(field.type, t.Struct):
                self._all_new(field.type, path + [name])
            else:
                self.changes.append(
                    FieldAdded(
                        tuple(path + [name]),
                        field.type,
                        field.default,
                        field.has_default,
                    )
                )

    def _all_gone(self, struct: t.Struct, path: typing.List[str]) -> None:
        for name, field in struct.fields.items():
            if isinstance(field.type, t.Struct):
                self._all_gone(field.type, path + [name])
            else:
                self.changes.append(
                    FieldRemoved(
                        tuple(path + [name]),
                        field.type,
                        field.default,
                        field.has_default,
                    )
                )


def name_similarity(field1: str, field2: str) -> float:
    # Special cases
    # TODO: fold these into tweaked levenshtein with different costs for different edits
    if field1 == field2:
        return 1.1
    if field1.replace("_", "") == field2.replace("_", ""):
        return 1.0
    if field1 == "".join(
        [word[0:1] for word in field2.split("_")]
    ) or field2 == "".join([word[0:1] for word in field1.split("_")]):
        return 1.0
    return 1 - levenshtein(field1, field2) / max(len(field1), len(field2))


def levenshtein(s1: str, s2: str) -> float:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row: List[float] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1.0]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitution_cost = 1.0
            if c1 == c2:
                substitution_cost = 0
            elif c1.lower() == c2.lower():
                substitution_cost = 0.25
            substitutions = previous_row[j] + substitution_cost
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _dump_schema(filename: str, type: typing.Type[Versioned]) -> None:
    serialized = pyron.to_string(materialize_type(type))
    with open(filename, "w") as f:
        f.write(serialized)


def _upgrade_schema(filename: str, config_clz: typing.Type[Versioned]) -> None:
    schema = load_schema(filename)
    assert isinstance(schema, t.Struct)
    checker = SchemaChecker(schema, config_clz)
    if checker.severity() >= Severity.WARN:
        checker.print_report()
    else:
        _dump_schema(filename, config_clz)
        click.secho("Schema updated", fg="green")


def _upgrade_config(
    filename: str,
    config_clz: typing.Type[Versioned],
    elide_defaults: bool,
    dry_run: bool,
) -> None:
    click.secho(filename, fg="cyan")
    config, schedules = _typed_load(config_clz, Path(filename))
    upgraded_config = _typed_dump(
        config, elide_defaults=elide_defaults, schedules=schedules
    )
    if dry_run:
        original_config = open(filename).read()
        diff = difflib.unified_diff(
            original_config.splitlines(),
            upgraded_config.splitlines(),
        )
        for line in list(diff)[3:]:
            if line.startswith("+"):
                click.secho(line, fg="green")
            elif line.startswith("-"):
                click.secho(line, fg="red")
            elif line.startswith("^"):
                click.secho(line, fg="blue")
            else:
                click.echo(line)
    else:
        open(filename, "w").write(upgraded_config)


CONFIG_CLZ: typing.Type[Any] = None  # type: ignore


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def dump_schema(filename: str) -> None:
    global CONFIG_CLZ
    _dump_schema(filename, CONFIG_CLZ)


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def upgrade_schema(filename: str) -> None:
    global CONFIG_CLZ
    _upgrade_schema(filename, CONFIG_CLZ)


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def check_schema(filename: str) -> None:
    global CONFIG_CLZ
    old = load_schema(filename)
    assert isinstance(old, t.Struct)
    SchemaChecker(old, CONFIG_CLZ).print_report()


@cli.command()
@click.argument("files", nargs=-1, type=click.Path())
@click.option("--include-defaults", is_flag=True)
@click.option("--dry-run", is_flag=True)
def upgrade_config(
    files: typing.List[str], include_defaults: bool, dry_run: bool
) -> None:
    for file in files:
        _upgrade_config(file, CONFIG_CLZ, not include_defaults, dry_run)


def schema_evolution_cli(config_clz: typing.Type[Any]) -> None:
    global CONFIG_CLZ
    CONFIG_CLZ = config_clz
    cli()
