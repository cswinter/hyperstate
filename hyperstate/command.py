import argparse
from typing import Any, Callable, Dict, List, Type, TypeVar
import sys
import click
import hyperstate
from hyperstate.hyperstate import (
    FieldNotFoundError,
    FieldsNotFoundError,
    StateManager,
)
from hyperstate.serde import DeserializeTypeError, DeserializeValueError

T = TypeVar("T")
C = TypeVar("C")
S = TypeVar("S")


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
                raise
            return f(cfg)

        return _f

    return _command


def stateful_command(
    cfg_cls: Type[C],
    state_cls: Type[S],
    initial_state: Callable[[C, Dict[str, Any]], S],
    checkpoint_key: str = "step",
) -> Callable[[Callable[[StateManager[C, S]], T]], Callable[[], T]]:

    # Evaluate lazily to materialize type annotations
    def _command(f: Callable[[Any], T]) -> Callable[[], T]:
        def _f() -> T:
            args = _parse_args(stateful=True)
            if args.hps_info is not None:
                hyperstate.help(cfg_cls, args.hps_info)
                sys.exit(0)

            if args.config is not None and args.resume_from is not None:
                print(
                    click.style("error", fg="red")
                    + ": Cannot specify both --config and --resume-from."
                )
                sys.exit(1)

            _check_overrides(cfg_cls, args.hyperparams)

            sm = StateManager(
                cfg_cls,
                state_cls,
                initial_state=initial_state,
                init_path=args.config or args.resume_from,
                checkpoint_dir=args.checkpoint_dir,
                overrides=args.hyperparams,
                checkpoint_key=checkpoint_key,
            )

            try:
                return f(sm)
            except Exception as e:
                _print_config_exception(e, args.verbose)
                raise

        return _f

    return _command


def _check_overrides(cls: Type[C], overrides: List[str]) -> None:
    for override in overrides:
        if "=" not in override:
            print(
                click.style("error", fg="red")
                + f": Invalid override '{override}'. Expected format: 'field.name=value'."
            )
            print()
            print(f"Info for '{override}':")
            hyperstate.help(cls, override)
            sys.exit(0)


def _print_config_exception(e: Exception, verbose: bool) -> None:
    if isinstance(e, FieldsNotFoundError):
        print(click.style("error", fg="red") + ": " + str(e))
        for error in e.not_found_errors:
            print()
            print(f"Field most similar to '{error.field_name}':")
            hyperstate.help(error.cls, error.field_name)
        sys.exit(1)
    elif isinstance(e, FieldNotFoundError):
        print(click.style("error", fg="red") + ": " + str(e))
        print()
        print("Most similar fields:")
        hyperstate.help(e.cls, e.field_name)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    elif isinstance(e, DeserializeTypeError) or isinstance(e, DeserializeValueError):
        print(click.style("error", fg="red") + ": " + str(e))
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
