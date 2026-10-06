# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
from numpy.typing import NDArray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.adaptive_threshold import MACSThreshold, SCSThreshold  # type: ignore[import]


class PercentileThresholdStrategy:
    def __init__(
        self,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.0,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def calculate_thresholds(self, mse_scores: NDArray[np.float64]) -> tuple[float, float]:
        """Calculate lower and upper thresholds for anomaly detection.
        Args:
            mse_scores: Array of reconstruction MSE scores
        Returns:
            Tuple of (lower_threshold, upper_threshold)
        """
        lower_threshold = np.maximum(0, np.percentile(mse_scores, self.lower_percentile))
        p99 = np.percentile(mse_scores, self.upper_percentile)

        mean = np.mean(mse_scores)
        std = np.std(mse_scores)
        cv = std / mean if mean > 0 else 1.0

        adaptive_factor = 1.0 + 0.5 * min(
            cv, 1.0
        )  # cv contributes at most 1.5 to the adaptive factor, for when datasets are highly variable

        upper_threshold = p99 * adaptive_factor
        return np.float64(lower_threshold).item(), np.float64(upper_threshold).item()


class SCSThresholdStrategy:
    """Segmented Confidence Sequences (SCS) Thresholding Strategy.

    Adapts the SCS method from adaptive_threshold.py to work with MSE scores by segmenting them
    and calculating confidence bounds per segment for direct anomaly detection.
    """

    def __init__(
        self,
        window_size: int = 200,
        confidence_level: float = 0.99,
        n_segments: int = 3,
        segmentation_method: str = "apca",
        percentile_filter: float = 95.0,
        verbose: bool = False,  # Control logging output
    ) -> None:
        self.scs_thresholder = SCSThreshold(
            window_size=window_size,
            confidence_level=confidence_level,
            n_segments=n_segments,
            segmentation_method=segmentation_method,
            percentile_filter=percentile_filter,
            verbose=verbose,
        )


class MACSThresholdStrategy:
    """Moving Average with Confidence Sequences (MACS) Thresholding Strategy.

    Adapts the MACS method from adaptive_threshold.py to work with MSE scores by applying
    moving average smoothing and calculating confidence bounds for anomaly detection.
    """

    def __init__(
        self,
        short_window: int = 10,
        medium_window: int = 50,
        long_window: int = 250,
        confidence_level: float = 0.95,
        verbose: bool = False,  # Control logging output
    ) -> None:
        self.macs_thresholder = MACSThreshold(
            short_window=short_window,
            medium_window=medium_window,
            long_window=long_window,
            confidence_level=confidence_level,
            verbose=verbose,
        )
