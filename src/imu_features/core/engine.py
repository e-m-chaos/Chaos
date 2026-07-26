"""FeatureEngine: orchestrates feature extraction over an IMUWindow."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .registry import REGISTRY, FeatureRegistry, FeatureSpec
from .signal import IMUWindow


class FeatureEngine:
    """Computes a flat dict of named features from an IMUWindow.

    Selection is either by family (e.g. ``families=["statistical", "frequency"]``)
    or by explicit feature name/key (e.g. ``features=["mean", "frequency.dominant_frequency"]``).
    With neither argument, every registered feature is used.

    Output keys follow ``{sensor}_{channel}_{family}_{feature}`` for
    per-channel features (channel in x/y/z/mag) and
    ``{sensor}_triaxial_{family}_{feature}`` for whole-vector features.
    """

    def __init__(
        self,
        families: Optional[Sequence[str]] = None,
        features: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        registry: FeatureRegistry = REGISTRY,
    ) -> None:
        self.registry = registry
        self.selected: List[FeatureSpec] = self._resolve(families, features, exclude)
        if not self.selected:
            raise ValueError("no features matched the given families/features selection")

    def _resolve(
        self,
        families: Optional[Sequence[str]],
        features: Optional[Sequence[str]],
        exclude: Optional[Sequence[str]],
    ) -> List[FeatureSpec]:
        specs = self.registry.all()
        if features:
            wanted = set(features)
            specs = [s for s in specs if s.key in wanted or s.name in wanted]
        elif families:
            wanted = set(families)
            specs = [s for s in specs if s.family in wanted]
        if exclude:
            excluded = set(exclude)
            specs = [s for s in specs if s.key not in excluded and s.name not in excluded]
        return specs

    def extract(self, window: IMUWindow) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sensor_name, arr in window.sensors().items():
            channels = {
                "x": arr[:, 0],
                "y": arr[:, 1],
                "z": arr[:, 2],
                "mag": window.magnitude(sensor_name),
            }
            for spec in self.selected:
                if spec.scope == "channel":
                    for ch_name, ch_data in channels.items():
                        key = f"{sensor_name}_{ch_name}_{spec.family}_{spec.name}"
                        out[key] = self._call(spec, ch_data, window.sample_rate)
                else:  # triaxial
                    key = f"{sensor_name}_triaxial_{spec.family}_{spec.name}"
                    out[key] = self._call(spec, arr, window.sample_rate)
        return out

    @staticmethod
    def _call(spec: FeatureSpec, data: np.ndarray, sample_rate: float) -> float:
        if len(data) < spec.min_samples:
            return float("nan")
        return float(spec.func(data, sample_rate))

    def extract_many(self, windows: Iterable[IMUWindow]):
        """Extract features for every window. Returns a pandas.DataFrame if
        pandas is installed, otherwise a list of dicts."""
        rows = [self.extract(w) for w in windows]
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            return rows

    def feature_names(self, window: IMUWindow) -> List[str]:
        return list(self.extract(window).keys())
