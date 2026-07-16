from __future__ import annotations

from typing import TypedDict


class EvalConfig(TypedDict):
    """Fully resolved configuration used internally by ``Evaluator``.

    Every field is guaranteed to be present after
    ``Evaluator.get_default_eval_config`` values have been merged in.
    """

    USE_PARALLEL: bool
    NUM_PARALLEL_CORES: int
    BREAK_ON_ERROR: bool
    RETURN_ON_ERROR: bool
    LOG_ON_ERROR: str | None
    PRINT_RESULTS: bool
    PRINT_ONLY_COMBINED: bool
    PRINT_CONFIG: bool
    TIME_PROGRESS: bool
    DISPLAY_LESS_PROGRESS: bool
    OUTPUT_SUMMARY: bool
    OUTPUT_EMPTY_CLASSES: bool
    OUTPUT_DETAILED: bool
    PLOT_CURVES: bool


class EvalConfigInput(TypedDict, total=False):
    """Partial configuration accepted by ``Evaluator.__init__``.

    Any field left unset falls back to the value returned by
    ``Evaluator.get_default_eval_config``.
    """

    USE_PARALLEL: bool
    NUM_PARALLEL_CORES: int
    BREAK_ON_ERROR: bool
    RETURN_ON_ERROR: bool
    LOG_ON_ERROR: str | None
    PRINT_RESULTS: bool
    PRINT_ONLY_COMBINED: bool
    PRINT_CONFIG: bool
    TIME_PROGRESS: bool
    DISPLAY_LESS_PROGRESS: bool
    OUTPUT_SUMMARY: bool
    OUTPUT_EMPTY_CLASSES: bool
    OUTPUT_DETAILED: bool
    PLOT_CURVES: bool
