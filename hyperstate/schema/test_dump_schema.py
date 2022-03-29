from dataclasses import dataclass
from typing import Dict, List, Literal
import hyperstate
import pyron

from hyperstate.schema.types import materialize_type


@dataclass
class SubSubConfig:
    a: int
    b: str
    c: Literal["x", "y", 5]


@dataclass
class SubConfig:
    cs: List[SubSubConfig]
    f: float


@dataclass
class Config(hyperstate.Versioned):
    subconfig: SubConfig

    @classmethod
    def version(clz) -> int:
        return 1


def test_dump_schema() -> None:
    print(pyron.to_string(materialize_type(Config)))
