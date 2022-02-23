from dataclasses import dataclass
import enum
from typing import Any, Dict, List, Optional
import typing
import tempfile
from hyperstate.hyperstate import dumps, loads
from hyperstate.schema.rewrite_rule import (
    AddDefault,
    ChangeDefault,
    CheckValue,
    DeleteField,
    MapFieldValue,
    RenameField,
    RewriteRule,
)
from hyperstate.schema.schema_change import (
    DefaultValueChanged,
    EnumVariantAdded,
    EnumVariantRemoved,
    EnumVariantRenamed,
    EnumVariantValueChanged,
    FieldAdded,
    FieldRemoved,
    FieldRenamed,
    SchemaChange,
    DefaultValueRemoved,
    TypeChanged,
)

from hyperstate.schema.schema_checker import (
    SchemaChecker,
    Severity,
)
from hyperstate.schema.types import (
    Enum,
    Struct,
    materialize_type,
    Type,
    Primitive,
    Option,
    Literal,
)
from hyperstate.schema.versioned import Versioned


@dataclass
class ConfigV1(Versioned):
    steps: int
    learning_rate: float
    batch_size: int
    epochs: int

    @classmethod
    def version(clz) -> int:
        return 1


@dataclass
class ConfigV2Error(ConfigV1):
    optimizer: str

    @classmethod
    def version(clz) -> int:
        return 2


@dataclass
class ConfigV2Warn(ConfigV1):
    optimizer: Optional[str]

    @classmethod
    def version(clz) -> int:
        return 2


@dataclass
class ConfigV2Info(ConfigV1):
    optimizer: str = "sgd"

    @classmethod
    def version(clz) -> int:
        return 2


def test_config_v1_to_v2() -> None:
    check_schema(
        ConfigV1,
        ConfigV2Info,
        [
            FieldAdded(
                field=("optimizer",),
                type=Primitive("str"),
                has_default=True,
                default="sgd",
            )
        ],
        [],
        Severity.INFO,
    )
    check_schema(
        ConfigV1,
        ConfigV2Warn,
        [
            FieldAdded(
                ("optimizer",),
                type=Option(Primitive(type="str")),
                has_default=False,
                default=None,
            )
        ],
        [AddDefault(field=("optimizer",), default=None)],
        Severity.WARN,
    )
    check_schema(
        ConfigV1,
        ConfigV2Error,
        [
            FieldAdded(
                ("optimizer",),
                type=Primitive(type="str"),
            )
        ],
        [],
        Severity.ERROR,
    )
    automatic_upgrade(
        ConfigV1(
            steps=1,
            learning_rate=0.1,
            batch_size=32,
            epochs=10,
        ),
        ConfigV2Warn(
            steps=1,
            learning_rate=0.1,
            batch_size=32,
            epochs=10,
            optimizer=None,
        ),
    )
    automatic_upgrade(
        ConfigV1(
            steps=1,
            learning_rate=0.1,
            batch_size=32,
            epochs=10,
        ),
        ConfigV2Info(
            steps=1,
            learning_rate=0.1,
            batch_size=32,
            epochs=10,
            optimizer="sgd",
        ),
    )


@dataclass
class ConfigV3(Versioned):
    steps: int
    lr: float
    batch_size: int
    epochs: int
    optimizer: typing.Literal["adam", "sgd"] = "adam"

    @classmethod
    def version(clz) -> int:
        return 3

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        return {
            2: [
                CheckValue(field=("optimizer",), allowed_values={"adam", "sgd"}),
                ChangeDefault(field=("optimizer",), new_default="adam"),
                RenameField(old_field=("learning_rate",), new_field=("lr",)),
            ],
        }


def test_config_v2_to_v3() -> None:
    check_schema(
        ConfigV2Info,
        ConfigV3,
        [
            TypeChanged(
                field=("optimizer",),
                old=Primitive("str"),
                new=Literal(allowed_values={"adam", "sgd"}),
            ),
            DefaultValueChanged(field=("optimizer",), old="sgd", new="adam"),
            FieldRenamed(field=("learning_rate",), new_name=("lr",)),
        ],
        [
            CheckValue(field=("optimizer",), allowed_values={"adam", "sgd"}),
            ChangeDefault(field=("optimizer",), new_default="adam"),
            RenameField(old_field=("learning_rate",), new_field=("lr",)),
        ],
        Severity.WARN,
    )
    automatic_upgrade(
        ConfigV2Info(
            steps=1,
            learning_rate=0.1,
            batch_size=32,
            epochs=10,
        ),
        ConfigV3(
            steps=1,
            lr=0.1,
            batch_size=32,
            epochs=10,
            optimizer="sgd",
        ),
    )


@dataclass
class OptimizerConfig:
    lr: float
    batch_size: int
    optimizer: typing.Literal["adam", "sgd"] = "adam"


class TaskType(enum.Enum):
    COINRUN = "CoinRun"
    STARPILOT = "StarPilot"
    MAZE = "MAZE"


@dataclass
class TaskConfig:
    task_type: TaskType = TaskType.COINRUN
    difficulty: int = 1


@dataclass
class ConfigV4(Versioned):
    steps: int
    epochs: int
    optimizer: OptimizerConfig
    task: TaskConfig

    @classmethod
    def version(clz) -> int:
        return 4


def test_config_v3_to_v4() -> None:
    check_schema(
        ConfigV3,
        ConfigV4,
        [
            FieldAdded(
                field=("task", "task_type"),
                type=Enum(
                    name="TaskType",
                    variants={
                        "COINRUN": "CoinRun",
                        "STARPILOT": "StarPilot",
                        "MAZE": "MAZE",
                    },
                ),
                default="CoinRun",
                has_default=True,
            ),
            FieldAdded(
                field=("task", "difficulty"),
                type=Primitive("int"),
                default=1,
                has_default=True,
            ),
            FieldRenamed(field=("optimizer",), new_name=("optimizer", "optimizer")),
            FieldRenamed(field=("lr",), new_name=("optimizer", "lr")),
            FieldRenamed(field=("batch_size",), new_name=("optimizer", "batch_size")),
        ],
        [
            RenameField(old_field=("optimizer",), new_field=("optimizer", "optimizer")),
            RenameField(old_field=("lr",), new_field=("optimizer", "lr")),
            RenameField(
                old_field=("batch_size",), new_field=("optimizer", "batch_size")
            ),
        ],
        Severity.WARN,
    )
    automatic_upgrade(
        ConfigV3(
            steps=1,
            lr=0.1,
            batch_size=32,
            epochs=10,
            optimizer="sgd",
        ),
        ConfigV4(
            steps=1,
            epochs=10,
            optimizer=OptimizerConfig(lr=0.1, batch_size=32, optimizer="sgd"),
            task=TaskConfig(),
        ),
    )


def test_config_v4_to_v3() -> None:
    check_schema(
        ConfigV4,
        ConfigV3,
        [
            FieldRemoved(
                field=("task", "task_type"),
                type=Enum(
                    name="TaskType",
                    variants={
                        "COINRUN": "CoinRun",
                        "STARPILOT": "StarPilot",
                        "MAZE": "MAZE",
                    },
                ),
                default="CoinRun",
                has_default=True,
            ),
            FieldRemoved(
                field=("task", "difficulty"),
                type=Primitive("int"),
                default=1,
                has_default=True,
            ),
            FieldRenamed(field=("optimizer", "lr"), new_name=("lr",)),
            FieldRenamed(field=("optimizer", "batch_size"), new_name=("batch_size",)),
            FieldRenamed(field=("optimizer", "optimizer"), new_name=("optimizer",)),
        ],
        [
            DeleteField(field=("task", "task_type")),
            DeleteField(field=("task", "difficulty")),
            RenameField(old_field=("optimizer", "lr"), new_field=("lr",)),
            RenameField(
                old_field=("optimizer", "batch_size"), new_field=("batch_size",)
            ),
            RenameField(old_field=("optimizer", "optimizer"), new_field=("optimizer",)),
        ],
        Severity.WARN,
    )


class ChangedTaskType(enum.Enum):
    CR = "CoinRun"
    StarPilot = "StarPilot"
    MAZE = "Maze"
    MINER = "Miner"


@dataclass
class ChangedTaskConfig:
    task_type: ChangedTaskType = ChangedTaskType.CR
    difficulty: int = 1


@dataclass
class ConfigV5(Versioned):
    epochs: int
    optimizer: OptimizerConfig
    task: ChangedTaskConfig
    steps: int = 10

    @classmethod
    def version(clz) -> int:
        return 5


def test_config_v4_to_v5() -> None:
    check_schema(
        ConfigV4,
        ConfigV5,
        [
            EnumVariantValueChanged(
                field=("task", "task_type"),
                enum_name="ChangedTaskType",
                variant="MAZE",
                old_value="MAZE",
                new_value="Maze",
            ),
            EnumVariantAdded(
                field=("task", "task_type"),
                enum_name="ChangedTaskType",
                variant="MINER",
                variant_value="Miner",
            ),
            EnumVariantRenamed(
                field=("task", "task_type"),
                enum_name="ChangedTaskType",
                old_variant_name="STARPILOT",
                new_variant_name="StarPilot",
            ),
            EnumVariantRenamed(
                field=("task", "task_type"),
                enum_name="ChangedTaskType",
                old_variant_name="COINRUN",
                new_variant_name="CR",
            ),
        ],
        [
            MapFieldValue(
                field=("task", "task_type"),
                map_fn=None,  # type: ignore
                rendered="lambda x: x if x != 'MAZE' else 'Maze'",
            ),
        ],
        Severity.WARN,
    )
    automatic_upgrade(
        ConfigV4(
            epochs=10,
            optimizer=OptimizerConfig(lr=0.1, batch_size=32, optimizer="sgd"),
            task=TaskConfig(task_type=TaskType.MAZE, difficulty=1),
            steps=10,
        ),
        ConfigV5(
            epochs=10,
            optimizer=OptimizerConfig(lr=0.1, batch_size=32, optimizer="sgd"),
            task=ChangedTaskConfig(task_type=ChangedTaskType.MAZE, difficulty=1),
            steps=10,
        ),
    )


class TaskTypeV6(enum.Enum):
    CR = "CoinRun"
    MAZE = "Maze"
    MINER = "Miner"


@dataclass
class TaskConfigV6:
    task_type: TaskTypeV6 = TaskTypeV6.CR
    difficulty: int = 1


@dataclass
class ConfigV6(Versioned):
    steps: float
    epochs: str
    optimizer: OptimizerConfig
    task: TaskConfigV6

    @classmethod
    def version(clz) -> int:
        return 6


def test_config_v5_to_v6() -> None:
    check_schema(
        ConfigV5,
        ConfigV6,
        [
            TypeChanged(field=("steps",), old=Primitive("int"), new=Primitive("float")),
            DefaultValueRemoved(field=("steps",), old=10),
            TypeChanged(field=("epochs",), old=Primitive("int"), new=Primitive("str")),
            EnumVariantRemoved(
                field=("task", "task_type"),
                enum_name="TaskTypeV6",
                variant="StarPilot",
                variant_value="StarPilot",
            ),
        ],
        [
            AddDefault(field=("steps",), default=10),
        ],
        Severity.ERROR,
    )


def test_serde_upgrade() -> None:
    config_v2 = ConfigV2Info(steps=1, learning_rate=0.1, batch_size=32, epochs=10)
    serialized = dumps(config_v2)
    config_v3 = loads(ConfigV3, serialized)
    assert config_v3 == ConfigV3(
        steps=1, lr=0.1, batch_size=32, epochs=10, optimizer="sgd"
    )


def check_schema(
    old: Any,
    new: Any,
    expected_changes: List[SchemaChange],
    expected_fixes: List[RewriteRule],
    expected_severity: Severity = Severity.ERROR,
    print_report: bool = False,
) -> None:
    with tempfile.TemporaryFile() as f:
        old_type = materialize_type(old)
        assert isinstance(old_type, Struct)
        checker = SchemaChecker(old_type, new, perform_upgrade=False)
        if print_report:
            checker.print_report()
        checker.proposed_fixes = [erase_lambdas(fix) for fix in checker.proposed_fixes]
        assert checker.changes == expected_changes
        assert checker.proposed_fixes == expected_fixes
        assert checker.severity() == expected_severity


def erase_lambdas(rule: RewriteRule) -> RewriteRule:
    if isinstance(rule, MapFieldValue):
        return MapFieldValue(
            field=rule.field,
            map_fn=None,  # type: ignore
            rendered=rule.rendered,
        )
    return rule


def automatic_upgrade(old: Any, new: Any) -> None:
    old_type = materialize_type(old.__class__)
    assert isinstance(old_type, Struct)
    autofixes = SchemaChecker(
        old_type, new.__class__, perform_upgrade=False
    ).proposed_fixes

    @dataclass
    class NewWithUpgradeRules(new.__class__):  # type: ignore
        @classmethod
        def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
            return {
                old.version(): autofixes,
            }

    checker = SchemaChecker(old_type, NewWithUpgradeRules, perform_upgrade=True)
    if checker.severity() >= Severity.WARN:
        checker.print_report()
    assert checker.severity() == Severity.INFO

    serialized = dumps(old)
    new_with_upgrade_rules = loads(NewWithUpgradeRules, serialized)
    assert new_with_upgrade_rules == NewWithUpgradeRules(**new.__dict__)
