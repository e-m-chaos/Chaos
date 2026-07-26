"""Feature registry: the plug-in mechanism that turns plain functions into
named, discoverable feature entries the engine can look up by family.

A feature function has the signature ``func(data, sample_rate) -> float``.
``data`` is a 1-D array for ``scope="channel"`` features (applied to each of
x/y/z and the resultant magnitude in turn) or an (n_samples, 3) array for
``scope="triaxial"`` features (applied once per sensor, using all three axes
jointly — e.g. PCA shape descriptors, cross-axis correlation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal

Scope = Literal["channel", "triaxial"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    scope: Scope
    func: Callable
    min_samples: int = 1
    description: str = ""

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
    ):
        def decorator(func: Callable) -> Callable:
            spec = FeatureSpec(
                name=name,
                family=family,
                scope=scope,
                func=func,
                min_samples=min_samples,
                description=description,
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
):
    """Decorator registering a feature function against the global registry."""
    return REGISTRY.register(name, family, scope, min_samples, description)
