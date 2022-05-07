from collections import namedtuple
from typing import Any


def remove_field(value: Any, field: str) -> Any:
    value = namedtuple(
        value.__class__.__name__,
        [f for f in value._fields if f is not field],
    )(*[getattr(value, f) for f in value._fields if f is not field])
