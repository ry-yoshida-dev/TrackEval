from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class EvalMetricProtocol(Protocol):
    """Interface required by ``Evaluator.evaluate`` from a metric instance."""

    @property
    def fields(self) -> Sequence[str]:
        """Return the names of the result fields produced by this metric."""
        ...

    def get_name(self) -> str:
        """Return the metric name used in evaluation output."""
        ...

    def eval_sequence(self, data: Mapping[str, object], /) -> Mapping[str, object]:
        """Evaluate the metric for a single sequence."""
        ...

    def combine_sequences(self, all_res: Mapping[str, object], /) -> Mapping[str, object]:
        """Combine per-sequence results into a single result."""
        ...

    def combine_classes_class_averaged(
        self,
        all_res: Mapping[str, object],
        ignore_empty_classes: bool = False,
        /,
    ) -> Mapping[str, object]:
        """Combine per-class results by averaging over classes."""
        ...

    def combine_classes_det_averaged(self, all_res: Mapping[str, object], /) -> Mapping[str, object]:
        """Combine per-class results by averaging over detections."""
        ...

    def print_table(self, table_res: Mapping[str, object], tracker: str, cls: str, /) -> None:
        """Print a table of results for all sequences."""
        ...

    def summary_results(self, table_res: Mapping[str, object], /) -> Mapping[str, object]:
        """Return a simple summary of final results for a tracker."""
        ...

    def detailed_results(self, table_res: Mapping[str, object], /) -> Mapping[str, object]:
        """Return detailed final results for a tracker."""
        ...

    def plot_single_tracker_results(
        self,
        table_res: Mapping[str, object],
        tracker: str,
        cls: str,
        output_folder: str,
        /,
    ) -> None:
        """Plot results of the metric for a single tracker."""
        ...
