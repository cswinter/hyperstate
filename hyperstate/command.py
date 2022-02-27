import argparse
from typing import Any, Callable, Type, TypeVar
import sys
import click
import hyperstate
from hyperstate.hyperstate import FieldNotFoundError, FieldsNotFoundError

T = TypeVar("T")
C = TypeVar("C")


def command(cls: Type[C]) -> Callable[[Callable[[C], T]], Callable[[], T]]:

    # Evaluate lazily to materialize type annotations
    def _command(f: Callable[[Any], T]) -> Callable[[], T]:
        def _f() -> T:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--config",
                type=str,
                default=None,
                help="Optional path to ron config file.",
            )
            parser.add_argument(
                "--hps-info",
                nargs="?",
                const="",
                help="Print information about hyperparameters.",
            )
            parser.add_argument(
                "-v",
                "--verbose",
                action="store_true",
                help="Print verbose output.",
            )
            parser.add_argument(
                dest="hyperparams",
                nargs="*",
                help="Hyperparameter values in the form of 'hyper.param=value'.",
            )
            args = parser.parse_args()
            if args.hps_info is not None:
                hyperstate.help(cls, args.hps_info)
                sys.exit(0)
            else:
                try:
                    for override in args.hyperparams:
                        if "=" not in override:
                            print(
                                click.style("ERROR", fg="red")
                                + f": Invalid override '{override}'. Expected format: 'field.name=value'."
                            )
                            print()
                            print(f"Info for '{override}':")
                            hyperstate.help(cls, override)
                            sys.exit(0)
                    cfg = hyperstate.load(
                        cls, file=args.config, overrides=args.hyperparams
                    )
                except FieldsNotFoundError as e:
                    print(click.style("ERROR", fg="red") + ": " + str(e))
                    for error in e.not_found_errors:
                        print()
                        print(f"Field most similar to '{error.field_name}':")
                        hyperstate.help(cls, error.field_name)
                    sys.exit(1)
                except FieldNotFoundError as e:
                    print(click.style("ERROR", fg="red") + ": " + str(e))
                    print()
                    print("Most similar fields:")
                    hyperstate.help(cls, e.field_name)
                    if args.verbose:
                        # Print traceback
                        import traceback

                        traceback.print_exc()
                    sys.exit(1)
                except (TypeError, ValueError) as e:
                    print(click.style("ERROR", fg="red") + ": " + str(e))
                    if args.verbose:
                        # Print traceback
                        import traceback

                        traceback.print_exc()
                    sys.exit(1)
            return f(cfg)

        return _f

    return _command
