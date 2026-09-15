from __future__ import annotations

import os
from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._base_dataset import _BaseDataset
from .ai_city_challenge_2024_layout import AICityChallenge2024Layout
from .. import _timing
from .. import utils
from ..utils import TrackEvalException

CAMERA_COLUMN = 0
OBJECT_COLUMN = 1
FRAME_COLUMN = 2
BOX_COLUMN_START = 3
BOX_COLUMN_STOP = 7
NUM_COLUMNS = 9
PEDESTRIAN_CLASS = "pedestrian"


class AICityChallenge2024(_BaseDataset):
    """
    Dataset class for the AI City Challenge 2024 multi-camera people tracking track.

    Ground truth and predictions share one whitespace-separated row layout::

        <camera_id> <object_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>

    Object ids are global, meaning one id denotes the same person in every
    camera of a scene. A scene is evaluated as a single sequence whose
    timestep axis is the flattening of its ``(camera, frame)`` pairs, so
    boxes only ever compete against boxes of the same camera while
    cross-camera id consistency is scored by the association terms of HOTA
    and Identity.

    The world coordinate columns are read but not used for matching, which
    is done with 2D box IoU in image space.
    """

    @staticmethod
    def get_default_dataset_config() -> dict[str, object]:
        """
        Default configuration values for the dataset.

        Returns
        -------
        dict[str, object]
            Configuration defaults, overridable by the config passed to
            ``__init__``.
        """
        code_path = utils.get_code_path()
        default_config: dict[str, object] = {
            "GT_FOLDER": os.path.join(code_path, "data/gt/ai_city_challenge_2024/"),
            "TRACKERS_FOLDER": os.path.join(
                code_path, "data/trackers/ai_city_challenge_2024/"
            ),
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": None,
            "CLASSES_TO_EVAL": [PEDESTRIAN_CLASS],
            "SPLIT_TO_EVAL": "test",
            "PRINT_CONFIG": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "TRACKER_DISPLAY_NAMES": None,
            "SEQ_INFO": None,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "GT_LOC_MAP": None,
            "SKIP_SPLIT_FOL": False,
        }
        return default_config

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        """
        Initialise the dataset and validate that every required file exists.

        Parameters
        ----------
        config : Mapping[str, object] | None
            Configuration overrides; missing keys fall back to
            ``get_default_dataset_config``.

        Raises
        ------
        TrackEvalException
            If an invalid class is requested, no sequence is selected, or a
            ground-truth or tracker file is missing.
        """
        super().__init__()
        self.config = utils.init_config(
            config, self.get_default_dataset_config(), self.get_name()
        )

        split_folder = (
            "" if self.config["SKIP_SPLIT_FOL"] else str(self.config["SPLIT_TO_EVAL"])
        )
        self.gt_fol = os.path.join(str(self.config["GT_FOLDER"]), split_folder)
        self.tracker_fol = os.path.join(
            str(self.config["TRACKERS_FOLDER"]), split_folder
        )
        self.should_classes_combine = False
        self.use_super_categories = False
        self.super_categories = {}

        output_folder = self.config["OUTPUT_FOLDER"]
        self.output_fol = (
            self.tracker_fol if output_folder is None else str(output_folder)
        )
        self.tracker_sub_fol = str(self.config["TRACKER_SUB_FOLDER"])
        self.output_sub_fol = str(self.config["OUTPUT_SUB_FOLDER"])

        self.valid_classes = [PEDESTRIAN_CLASS]
        requested_classes = self.config["CLASSES_TO_EVAL"]
        if not isinstance(requested_classes, (list, tuple)):
            raise TrackEvalException("CLASSES_TO_EVAL must be a list of class names.")
        self.class_list = [str(name).lower() for name in requested_classes]
        invalid_classes = [
            name for name in self.class_list if name not in self.valid_classes
        ]
        if invalid_classes:
            raise TrackEvalException(
                "Attempted to evaluate an invalid class. Only the "
                + f"{PEDESTRIAN_CLASS} class is valid, got: "
                + ", ".join(invalid_classes)
            )

        self.seq_list = self._resolve_seq_list()
        if len(self.seq_list) < 1:
            raise TrackEvalException("No sequences are selected to be evaluated.")

        self.seq_layouts = {seq: self._build_layout(seq) for seq in self.seq_list}
        self.seq_lengths = {
            seq: layout.num_timesteps for seq, layout in self.seq_layouts.items()
        }

        self.tracker_list = self._resolve_tracker_list()
        self.tracker_to_disp = self._resolve_tracker_display_names()
        self._validate_tracker_files()

    def _resolve_seq_list(self) -> list[str]:
        """
        Determine which scenes to evaluate.

        ``SEQ_INFO`` keys take precedence; otherwise every sub-directory of
        the ground-truth folder holding a ground-truth file is used.

        Returns
        -------
        list[str]
            Scene names to evaluate.

        Raises
        ------
        TrackEvalException
            If the ground-truth folder does not exist.
        """
        seq_info = self.config["SEQ_INFO"]
        if isinstance(seq_info, Mapping):
            return [str(seq) for seq in seq_info]

        if not os.path.isdir(self.gt_fol):
            raise TrackEvalException(f"GT folder not found: {self.gt_fol}")
        return sorted(
            entry
            for entry in os.listdir(self.gt_fol)
            if os.path.isfile(self._gt_file(entry))
        )

    def _resolve_tracker_list(self) -> list[str]:
        """
        Determine which trackers to evaluate.

        Returns
        -------
        list[str]
            Tracker folder names to evaluate.

        Raises
        ------
        TrackEvalException
            If trackers must be discovered but the tracker folder is absent.
        """
        trackers_to_eval = self.config["TRACKERS_TO_EVAL"]
        if trackers_to_eval is None:
            if not os.path.isdir(self.tracker_fol):
                raise TrackEvalException(
                    f"Trackers folder not found: {self.tracker_fol}"
                )
            return sorted(os.listdir(self.tracker_fol))
        if not isinstance(trackers_to_eval, (list, tuple)):
            raise TrackEvalException("TRACKERS_TO_EVAL must be a list of names.")
        return [str(tracker) for tracker in trackers_to_eval]

    def _resolve_tracker_display_names(self) -> dict[str, str]:
        """
        Pair every tracker with the name used in evaluation output.

        Returns
        -------
        dict[str, str]
            Mapping of tracker folder name to display name.

        Raises
        ------
        TrackEvalException
            If the supplied display names do not match the tracker count.
        """
        display_names = self.config["TRACKER_DISPLAY_NAMES"]
        if display_names is None:
            return dict(zip(self.tracker_list, self.tracker_list))
        if (
            not isinstance(display_names, (list, tuple))
            or len(display_names) != len(self.tracker_list)
        ):
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )
        return dict(
            zip(self.tracker_list, (str(name) for name in display_names))
        )

    def _validate_tracker_files(self) -> None:
        """
        Check that every tracker provides a prediction file per scene.

        Raises
        ------
        TrackEvalException
            If a prediction file is missing.
        """
        for tracker in self.tracker_list:
            for seq in self.seq_list:
                tracker_file = self._tracker_file(tracker, seq)
                if not os.path.isfile(tracker_file):
                    raise TrackEvalException(
                        f"Tracker file not found: {tracker_file}"
                    )

    def _gt_file(self, seq: str) -> str:
        """
        Path of the ground-truth file of a scene.

        ``GT_LOC_MAP`` gives the path of every scene explicitly and takes
        precedence over ``GT_LOC_FORMAT``, which allows ground-truth files that
        a single template cannot address.

        Parameters
        ----------
        seq : str
            Scene name.

        Returns
        -------
        str
            Ground-truth file path.

        Raises
        ------
        TrackEvalException
            If ``GT_LOC_MAP`` is given but has no entry for the scene.
        """
        gt_loc_map = self.config["GT_LOC_MAP"]
        if isinstance(gt_loc_map, Mapping):
            if seq not in gt_loc_map:
                raise TrackEvalException(
                    f"GT_LOC_MAP has no entry for sequence: {seq}"
                )
            return str(gt_loc_map[seq])
        return str(self.config["GT_LOC_FORMAT"]).format(
            gt_folder=self.gt_fol, seq=seq
        )

    def _tracker_file(self, tracker: str, seq: str) -> str:
        """
        Path of the prediction file of a tracker for a scene.

        Parameters
        ----------
        tracker : str
            Tracker folder name.
        seq : str
            Scene name.

        Returns
        -------
        str
            Prediction file path.
        """
        return os.path.join(
            self.tracker_fol, tracker, self.tracker_sub_fol, f"{seq}.txt"
        )

    def _build_layout(self, seq: str) -> AICityChallenge2024Layout:
        """
        Derive the timestep layout of a scene from its ground truth.

        The ground truth is authoritative for both the camera set and the
        frame count, so no sequence metadata file is needed.

        Parameters
        ----------
        seq : str
            Scene name.

        Returns
        -------
        AICityChallenge2024Layout
            Camera set and frame count of the scene.

        Raises
        ------
        TrackEvalException
            If the ground-truth file is missing or empty.
        """
        gt_file = self._gt_file(seq)
        if not os.path.isfile(gt_file):
            raise TrackEvalException(f"GT file not found for sequence {seq}: {gt_file}")
        if os.path.getsize(gt_file) == 0:
            raise TrackEvalException(f"GT file is empty for sequence {seq}: {gt_file}")

        columns = self._read_columns(gt_file, (CAMERA_COLUMN, FRAME_COLUMN))
        camera_ids = np.unique(columns[:, 0].astype(np.int64))
        largest_frame_id = int(columns[:, 1].max())
        return AICityChallenge2024Layout(
            camera_ids=camera_ids,
            num_frames=largest_frame_id + 1,
        )

    @staticmethod
    def _read_columns(
        file: str,
        usecols: tuple[int, ...],
    ) -> NDArray[np.float64]:
        """
        Read selected columns of a whitespace-separated file.

        Parameters
        ----------
        file : str
            File to read.
        usecols : tuple[int, ...]
            Column indices to keep.

        Returns
        -------
        NDArray[np.float64]
            Two-dimensional array holding the requested columns.

        Raises
        ------
        TrackEvalException
            If the file cannot be parsed as whitespace-separated numbers.
        """
        try:
            return np.loadtxt(file, dtype=np.float64, usecols=usecols, ndmin=2)
        except ValueError as error:
            raise TrackEvalException(
                f"Cannot parse {file} as the whitespace-separated AI City "
                + f"Challenge 2024 format: {error}"
            ) from error

    @staticmethod
    def _load_rows(file: str) -> NDArray[np.float64]:
        """
        Read every row of an AI City Challenge 2024 format file.

        Parameters
        ----------
        file : str
            File to read.

        Returns
        -------
        NDArray[np.float64]
            Array of shape ``(num_rows, num_columns)``; empty files yield
            zero rows.

        Raises
        ------
        TrackEvalException
            If the file is missing, unparsable, or has too few columns.
        """
        if not os.path.isfile(file):
            raise TrackEvalException(f"File not found: {file}")
        if os.path.getsize(file) == 0:
            return np.empty((0, NUM_COLUMNS), dtype=np.float64)

        try:
            rows = np.loadtxt(file, dtype=np.float64, ndmin=2)
        except ValueError as error:
            raise TrackEvalException(
                f"Cannot parse {file} as the whitespace-separated AI City "
                + f"Challenge 2024 format: {error}"
            ) from error

        if rows.shape[1] < NUM_COLUMNS:
            raise TrackEvalException(
                f"{file} has {rows.shape[1]} columns but the AI City Challenge "
                + f"2024 format needs at least {NUM_COLUMNS}. A comma-separated "
                + "MOTChallenge file placed here would produce this error."
            )
        return rows

    @staticmethod
    def _group_by_timestep(
        timesteps: NDArray[np.int64],
        num_timesteps: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """
        Order rows by timestep and locate each timestep's slice.

        Parameters
        ----------
        timesteps : NDArray[np.int64]
            Timestep index of every row.
        num_timesteps : int
            Timestep count of the sequence.

        Returns
        -------
        tuple[NDArray[np.int64], NDArray[np.int64]]
            Row order sorted by timestep, and the ``num_timesteps + 1``
            boundaries that slice that order per timestep.
        """
        order = np.argsort(timesteps, kind="stable").astype(np.int64)
        boundaries = np.searchsorted(
            timesteps[order], np.arange(num_timesteps + 1)
        ).astype(np.int64)
        return order, boundaries

    def _load_raw_file(
        self,
        tracker: str,
        seq: str,
        is_gt: bool,
    ) -> dict[str, object]:
        """
        Load ground-truth or prediction data for one scene.

        Parameters
        ----------
        tracker : str
            Tracker folder name, ignored when loading ground truth.
        seq : str
            Scene name.
        is_gt : bool
            Whether to load the ground truth rather than the predictions.

        Returns
        -------
        dict[str, object]
            Per-timestep ids and boxes under the ``gt_`` or ``tracker_``
            prefix, plus ``num_timesteps`` and ``seq``.
        """
        file = self._gt_file(seq) if is_gt else self._tracker_file(tracker, seq)
        layout = self.seq_layouts[seq]
        num_timesteps = layout.num_timesteps

        rows = self._load_rows(file)
        timesteps = layout.to_timesteps(
            camera_column=rows[:, CAMERA_COLUMN].astype(np.int64),
            frame_column=rows[:, FRAME_COLUMN].astype(np.int64),
            source=file,
        )
        object_ids = rows[:, OBJECT_COLUMN].astype(np.int64)
        boxes = rows[:, BOX_COLUMN_START:BOX_COLUMN_STOP]
        order, boundaries = self._group_by_timestep(timesteps, num_timesteps)

        ids_per_timestep: list[NDArray[np.int64]] = []
        dets_per_timestep: list[NDArray[np.float64]] = []
        confidences_per_timestep: list[NDArray[np.float64]] = []
        for timestep in range(num_timesteps):
            selection = order[boundaries[timestep] : boundaries[timestep + 1]]
            ids_per_timestep.append(object_ids[selection])
            dets_per_timestep.append(boxes[selection])
            if not is_gt:
                confidences_per_timestep.append(
                    np.ones(selection.size, dtype=np.float64)
                )

        prefix = "gt" if is_gt else "tracker"
        raw_data: dict[str, object] = {
            f"{prefix}_ids": ids_per_timestep,
            f"{prefix}_dets": dets_per_timestep,
            "num_timesteps": num_timesteps,
            "seq": seq,
        }
        if not is_gt:
            raw_data["tracker_confidences"] = confidences_per_timestep
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(
        self,
        raw_data: Mapping[str, object],
        cls: str,
    ) -> dict[str, object]:
        """
        Prepare a scene's data for evaluation.

        The format carries a single class, no distractor labels, no ignore
        regions and no detection confidences, so preprocessing only relabels
        ids to be contiguous and records the overview counts that metrics
        rely on.

        Parameters
        ----------
        raw_data : Mapping[str, object]
            Sequence data produced by ``get_raw_seq_data``.
        cls : str
            Class to evaluate.

        Returns
        -------
        dict[str, object]
            Data ready for the metrics.

        Raises
        ------
        TrackEvalException
            If a class other than the pedestrian class is requested.
        """
        if cls != PEDESTRIAN_CLASS:
            raise TrackEvalException(
                f"Evaluation is only valid for the {PEDESTRIAN_CLASS} class, "
                + f"got: {cls}"
            )
        self._check_unique_ids(raw_data)

        num_timesteps = cast(int, raw_data["num_timesteps"])
        gt_ids = cast("list[NDArray[np.int64]]", raw_data["gt_ids"])
        tracker_ids = cast("list[NDArray[np.int64]]", raw_data["tracker_ids"])
        gt_present_ids = self._collect_present_ids(gt_ids)
        tracker_present_ids = self._collect_present_ids(tracker_ids)

        data: dict[str, object] = {
            "gt_ids": self._relabel_ids(gt_ids, gt_present_ids),
            "tracker_ids": self._relabel_ids(tracker_ids, tracker_present_ids),
            "gt_dets": raw_data["gt_dets"],
            "tracker_dets": raw_data["tracker_dets"],
            "tracker_confidences": raw_data["tracker_confidences"],
            "similarity_scores": raw_data["similarity_scores"],
            "num_timesteps": num_timesteps,
            "seq": raw_data["seq"],
            "num_gt_ids": int(gt_present_ids.size),
            "num_tracker_ids": int(tracker_present_ids.size),
            "num_gt_dets": sum(int(ids.size) for ids in gt_ids),
            "num_tracker_dets": sum(int(ids.size) for ids in tracker_ids),
        }

        self._check_unique_ids(data, after_preproc=True)
        return data

    @staticmethod
    def _collect_present_ids(
        ids_per_timestep: list[NDArray[np.int64]],
    ) -> NDArray[np.int64]:
        """
        Collect the distinct ids of a sequence in ascending order.

        Parameters
        ----------
        ids_per_timestep : list[NDArray[np.int64]]
            Ids observed at every timestep.

        Returns
        -------
        NDArray[np.int64]
            Ascending distinct ids.
        """
        if not ids_per_timestep:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(ids_per_timestep))

    @staticmethod
    def _relabel_ids(
        ids_per_timestep: list[NDArray[np.int64]],
        present_ids: NDArray[np.int64],
    ) -> list[NDArray[np.int64]]:
        """
        Renumber ids to a contiguous range starting at zero.

        Metrics size their accumulators by the id count, so gaps in the
        original numbering would waste memory and break indexing.

        Parameters
        ----------
        ids_per_timestep : list[NDArray[np.int64]]
            Ids observed at every timestep.
        present_ids : NDArray[np.int64]
            Ascending distinct ids of the sequence.

        Returns
        -------
        list[NDArray[np.int64]]
            Ids renumbered to ``0..present_ids.size - 1``.
        """
        if present_ids.size == 0:
            return ids_per_timestep

        id_map = np.full(int(present_ids.max()) + 1, -1, dtype=np.int64)
        id_map[present_ids] = np.arange(present_ids.size, dtype=np.int64)
        return [id_map[ids] for ids in ids_per_timestep]

    def _calculate_similarities(
        self,
        gt_dets_t: object,
        tracker_dets_t: object,
    ) -> NDArray[np.float64]:
        """
        Score ground-truth boxes against tracker boxes of one timestep.

        Parameters
        ----------
        gt_dets_t : object
            Ground-truth boxes of the timestep in ``xywh`` form.
        tracker_dets_t : object
            Tracker boxes of the timestep in ``xywh`` form.

        Returns
        -------
        NDArray[np.float64]
            Pairwise intersection-over-union scores.
        """
        return self._calculate_box_ious(
            gt_dets_t, tracker_dets_t, box_format="xywh"
        )

    def get_display_name(self, tracker: str) -> str:
        """
        Name of a tracker as it should appear in evaluation output.

        Parameters
        ----------
        tracker : str
            Tracker folder name.

        Returns
        -------
        str
            Display name of the tracker.
        """
        return self.tracker_to_disp[tracker]
