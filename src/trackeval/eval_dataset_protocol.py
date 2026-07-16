from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class EvalDatasetProtocol(Protocol):
    """Interface required by ``Evaluator.evaluate`` from a dataset instance."""

    should_classes_combine: bool
    use_super_categories: bool
    super_categories: Mapping[str, Sequence[str]]

    def get_name(self) -> str:
        """Return the dataset name used in evaluation output."""
        ...

    def get_eval_info(self) -> tuple[list[str], list[str], list[str]]:
        """Return trackers, sequences, and classes to evaluate."""
        ...

    def get_output_fol(self, tracker: str, /) -> str:
        """Return the output folder for a given tracker."""
        ...

    def get_display_name(self, tracker: str, /) -> str:
        """Return the display name for a given tracker."""
        ...

    def get_raw_seq_data(self, tracker: str, seq: str, /) -> Mapping[str, object]:
        """Load raw tracker and ground-truth data for a single sequence."""
        ...

    def get_preprocessed_seq_data(
        self,
        raw_data: Mapping[str, object],
        cls: str,
        /,
    ) -> Mapping[str, object]:
        """Preprocess raw sequence data for a given class."""
        ...
