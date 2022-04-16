import argparse
from typing import Any, Callable, List, Type, TypeVar
import sys
import click
import hyperstate
from hyperstate.hyperstate import (
    FieldNotFoundError,
    FieldsNotFoundError,
    HyperState,
)

T = TypeVar("T")
C = TypeVar("C")
S = TypeVar("S")
HS = TypeVar("HS", bound=HyperState)


def _parse_args(stateful: bool) -> argparse.Namespace:
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
    if stateful:
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default=None,
            help="Checkpoints are persisted and restored from here.",
        )
        parser.add_argument(
            "--resume-from",
            type=str,
            default=None,
            help="Resume from a checkpoint. If --checkpoint-dir is set and contains a checkpoint, this is ignored.",
        )
    return parser.parse_args()


def command(cls: Type[C]) -> Callable[[Callable[[C], T]], Callable[[], T]]:

    # Evaluate lazily to materialize type annotations
    def _command(f: Callable[[C], T]) -> Callable[[], T]:
        def _f() -> T:
            args = _parse_args(stateful=False)
            if args.hps_info is not None:
                hyperstate.help(cls, args.hps_info)
                sys.exit(0)

            _check_overrides(cls, args.hyperparams)
            try:
                cfg = hyperstate.load(cls, file=args.config, overrides=args.hyperparams)
            except Exception as e:
                _print_config_exception(e, args.verbose)
            return f(cfg)

        return _f

    return _command


def stateful_command(
    cls: Type[HS],
    cfg_cls: Type[C],
    state_cls: Type[S],
) -> Callable[[Callable[[HS], T]], Callable[[], T]]:

    # Evaluate lazily to materialize type annotations
    def _command(f: Callable[[Any], T]) -> Callable[[], T]:
        def _f() -> T:
            args = _parse_args(stateful=True)
            if args.hps_info is not None:
                hyperstate.help(cls, args.hps_info)
                sys.exit(0)

            if args.config is not None and args.resume_from is not None:
                print(
                    click.style("ERROR", fg="red")
                    + ": Cannot specify both --config and --resume-from."
                )
                sys.exit(1)

            _check_overrides(cfg_cls, args.hyperparams)

            try:
                hs = cls(
                    cfg_cls,
                    state_cls,
                    args.config or args.resume_from,
                    args.checkpoint_dir,
                    args.hyperparams,
                )
            except Exception as e:
                _print_config_exception(e, args.verbose)
            return f(hs)

        return _f

    return _command


def _check_overrides(cls: Type[C], overrides: List[str]) -> None:
    for override in overrides:
        if "=" not in override:
            print(
                click.style("ERROR", fg="red")
                + f": Invalid override '{override}'. Expected format: 'field.name=value'."
            )
            print()
            print(f"Info for '{override}':")
            hyperstate.help(cls, override)
            sys.exit(0)


def _print_config_exception(e: Exception, verbose: bool) -> None:
    if isinstance(e, FieldsNotFoundError):
        print(click.style("ERROR", fg="red") + ": " + str(e))
        for error in e.not_found_errors:
            print()
            print(f"Field most similar to '{error.field_name}':")
            hyperstate.help(error.cls, error.field_name)
        sys.exit(1)
    elif isinstance(e, FieldNotFoundError):
        print(click.style("ERROR", fg="red") + ": " + str(e))
        print()
        print("Most similar fields:")
        hyperstate.help(e.cls, e.field_name)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    elif isinstance(e, TypeError) or isinstance(e, ValueError):
        print(click.style("ERROR", fg="red") + ": " + str(e))
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    else:
        raise e
