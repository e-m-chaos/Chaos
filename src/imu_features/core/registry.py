"""Feature registry: the plug-in mechanism that turns plain functions into
named, discoverable feature entries the engine can look up by family.

Most features have the signature ``func(data, sample_rate) -> float``.
``data`` is a 1-D array for ``scope="channel"`` features (applied to each of
x/y/z and the resultant magnitude in turn) or an (n_samples, 3) array for
``scope="triaxial"`` features (applied once per sensor, using all three axes
jointly — e.g. PCA shape descriptors, cross-axis correlation).

``scope="fusion"`` is different: these features genuinely need more than one
sensor at once (e.g. fusing accelerometer + gyroscope into an orientation
estimate), so they're applied once per window rather than once per sensor.
Their signature is ``func(window: IMUWindow) -> Dict[str, float]`` — they
return several named sub-values instead of one scalar — and they declare
which sensors they need via ``requires`` (e.g. ``("accel", "gyro")``); the
engine skips them for windows missing any required sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

Scope = Literal["channel", "triaxial", "fusion"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    scope: Scope
    func: Callable
    min_samples: int = 1
    description: str = ""
    requires: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        return f"{self.family}.{self.name}"


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: Dict[str, FeatureSpec] = {}

    def register(
        self,
        name: str,
        family: str,
        scope: Scope = "channel",
        min_samples: int = 1,
        description: str = "",
        requires: Tuple[str, ...] = (),
    ):
        def decorator(func: Callable) -> Callable:
            spec = FeatureSpec(
                name=name,
                family=family,
                scope=scope,
                func=func,
                min_samples=min_samples,
                description=description,
                requires=requires,
            )
            if spec.key in self._features:
                raise ValueError(f"feature '{spec.key}' is already registered")
            self._features[spec.key] = spec
            return func

        return decorator

    def get(self, key: str) -> FeatureSpec:
        return self._features[key]

    def families(self) -> List[str]:
        return sorted({spec.family for spec in self._features.values()})

    def by_family(self, family: str) -> List[FeatureSpec]:
        return [s for s in self._features.values() if s.family == family]

    def all(self) -> List[FeatureSpec]:
        return list(self._features.values())

    def __len__(self) -> int:
        return len(self._features)


REGISTRY = FeatureRegistry()


def register_feature(
    name: str,
    family: str,
    scope: Scope = "channel",
    min_samples: int = 1,
    description: str = "",
    requires: Tuple[str, ...] = (),
):
    """Decorator registering a feature function against the global registry."""
    return REGISTRY.register(name, family, scope, min_samples, description, requires)
