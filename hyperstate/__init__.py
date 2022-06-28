"""
Opinionated library for managing hyperparameters and mutable state of machine learning training systems. 
"""
import sys
import typing, typing_extensions

if sys.version_info < (3, 8):
    from typing_extensions import get_origin as get_origin_typing_extensions
    def get_origin_wrapped(t):
        if sys.version_info < (3, 8):
            return get_origin_typing_extensions(t)
        return typing.get_origin(t)
    setattr(typing, "Literal", typing_extensions.Literal)
    setattr(typing, "get_origin", get_origin_wrapped)
    setattr(typing, "get_args", typing_extensions.get_args)

from .hyperstate import StateManager, load, loads, dump, dumps
from .lazy import Serializable, Lazy, blob
from .schema.versioned import Versioned
from .schema.schema_checker import schema_evolution_cli
from .schema.help import help
from .command import command, stateful_command

__all__ = [
    "StateManager",
    "load",
    "loads",
    "dump",
    "dumps",
    "blob",
    "Serializable",
    "Lazy",
    "Versioned",
    "schema_evolution_cli",
    "help",
    "command",
    "stateful_command",
]
