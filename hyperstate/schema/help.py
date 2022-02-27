from typing import Any, List, Optional, Tuple, Type
from click import style, unstyle
from hyperstate.schema.schema_checker import name_similarity

from hyperstate.schema.types import _unwrap_container_type, materialize_type
import hyperstate.schema.types as t


def help(config_clz: Type[Any], query: str = "") -> None:
    config = materialize_type(config_clz)
    assert isinstance(config, t.Struct)
    if len(query) == 0:
        print_schema(config)
    else:
        fields = find_fields(config, query)
        fields = sorted(fields, key=lambda x: x[1], reverse=True)
        last_similarity = -1.0
        best_similarity = fields[0][1] if len(fields) > 0 else 0
        for i, (path, similarity, field) in enumerate(fields):
            if (
                similarity <= last_similarity
                and query not in field.name
                and similarity < 1.0
                and (
                    i > 3
                    or similarity < 0.4
                    or best_similarity >= 1.0
                    or best_similarity - similarity > 0.2
                )
            ):
                break
            unwrapped = _unwrap_container_type(field.type)
            if isinstance(unwrapped, t.Struct):
                line = (
                    style(".".join(path + [field.name]), fg="cyan")
                    + style(":", fg="white")
                    + " "
                    + style(str(field.type), fg="green")
                )
                if similarity >= 1.0:
                    print(line)
                    print_schema(unwrapped, depth=1, recurse=False)
                    last_similarity = similarity
                    continue
            else:
                line = (
                    style(".", fg="white").join(
                        [style(s, fg="cyan") for s in path + [field.name]]
                    )
                    + style(": ", fg="white")
                    + style(str(field.type), fg="green")
                )
                if field.default is not None:
                    line += style(" = ", fg="white") + style(
                        repr(field.default), fg="yellow"
                    )
            if field.docstring is not None:
                line += (
                    " " * (40 - len(unstyle(line)))
                    + style("  # ", fg="white")
                    + style(field.docstring, fg="white")
                )
            print(line)
            last_similarity = similarity


def print_schema(schema: t.Struct, depth: int = 0, recurse: bool = True) -> None:
    for f in schema.fields.values():
        line = "  " * depth + style(f.name, fg="cyan") + style(":", fg="white") + " "
        unwrapped = _unwrap_container_type(f.type)
        if isinstance(unwrapped, t.Struct):
            line += style(str(f.type), fg="green")
            print(line)
            line = ""
            if recurse:
                print_schema(unwrapped, depth + 1)
            continue
        else:
            line += style(f.type, fg="green")
        if f.default is not None:
            line += style(" = ", fg="white") + style(repr(f.default), fg="yellow")
        if line != "":
            if f.docstring is not None:
                line += (
                    " " * (40 - len(unstyle(line)))
                    + style("  # ", fg="white")
                    + style(f.docstring, fg="white")
                )
            print(line)


def find_fields(
    schema: t.Struct, query: str, path: Optional[List[str]] = None
) -> List[Tuple[List[str], float, t.Field]]:
    result = []
    for f in schema.fields.values():
        result.append((path or [], name_similarity(f.name, query), f))
        if isinstance(f.type, t.Struct):
            result.extend(find_fields(f.type, query, (path or []) + [f.name]))
    return result
