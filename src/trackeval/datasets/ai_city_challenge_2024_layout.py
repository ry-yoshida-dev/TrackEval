from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..utils import TrackEvalException


@dataclass(frozen=True)
class AICityChallenge2024Layout:
    """
    Flattening of a multi-camera scene onto a single TrackEval timestep axis.

    An AI City Challenge 2024 scene holds several cameras that share one
    global object-id space, while TrackEval evaluates a sequence as a flat
    list of timesteps. Every ``(camera, frame)`` pair therefore becomes its
    own timestep. Boxes from different cameras never share a timestep, so
    they are never matched against each other, whereas a global id reused
    consistently across cameras is rewarded by the association terms of
    HOTA and Identity.

    Attributes
    ----------
    camera_ids : NDArray[np.int64]
        Ascending camera ids that make up the scene.
    num_frames : int
        Frame count of every camera, taken as the largest ground-truth
        frame id plus one.
    """

    camera_ids: NDArray[np.int64]
    num_frames: int

    @property
    def num_cameras(self) -> int:
        """
        Camera count of the scene.

        Returns
        -------
        int
            Number of cameras the scene spans.
        """
        return int(self.camera_ids.size)

    @property
    def num_timesteps(self) -> int:
        """
        Timestep count of the flattened sequence.

        Returns
        -------
        int
            Product of the camera count and the per-camera frame count.
        """
        return self.num_cameras * self.num_frames

    def to_timesteps(
        self,
        camera_column: NDArray[np.int64],
        frame_column: NDArray[np.int64],
        source: str,
    ) -> NDArray[np.int64]:
        """
        Map ``(camera, frame)`` pairs onto flat timestep indices.

        Parameters
        ----------
        camera_column : NDArray[np.int64]
            Camera id of every row.
        frame_column : NDArray[np.int64]
            Frame id of every row.
        source : str
            File the rows were read from, quoted in error messages.

        Returns
        -------
        NDArray[np.int64]
            Timestep index of every row, ordered camera-major.

        Raises
        ------
        TrackEvalException
            If a camera id is absent from the scene, or a frame id falls
            outside the range established by the ground truth.
        """
        camera_indices = np.searchsorted(self.camera_ids, camera_column)
        is_within_bounds = camera_indices < self.camera_ids.size
        is_known_camera = np.zeros(camera_column.shape, dtype=bool)
        is_known_camera[is_within_bounds] = (
            self.camera_ids[camera_indices[is_within_bounds]]
            == camera_column[is_within_bounds]
        )
        if not bool(np.all(is_known_camera)):
            unknown_cameras = np.unique(camera_column[~is_known_camera])
            raise TrackEvalException(
                f"{source} refers to camera ids that the ground truth of this "
                + "scene does not contain: "
                + ", ".join(str(int(value)) for value in unknown_cameras)
            )

        is_valid_frame = (frame_column >= 0) & (frame_column < self.num_frames)
        if not bool(np.all(is_valid_frame)):
            invalid_frames = np.unique(frame_column[~is_valid_frame])
            raise TrackEvalException(
                f"{source} refers to frame ids outside the range "
                + f"0..{self.num_frames - 1} established by the ground truth: "
                + ", ".join(str(int(value)) for value in invalid_frames[:10])
            )

        return (
            camera_indices.astype(np.int64) * self.num_frames
            + frame_column.astype(np.int64)
        )
