# Architecture

`imu_features` is a small core (data model + registry + engine) around a
growing set of independent feature-family modules. The design goal is that
adding a new feature — or a whole new family — never requires touching the
engine, the registry, or any other family.

## Data model (`imu_features.core.signal.IMUWindow`)

An `IMUWindow` holds one fixed-length slice of synchronized samples for
whichever sensors are present (`accel`, `gyro`, `mag`), each shape
`(n_samples, 3)`, plus the window's `sample_rate`. It validates shape
consistency and equal length across sensors at construction time so every
downstream feature function can assume clean input.

`imu_features.core.windowing.segment_signal` turns a continuous multi-sensor
recording into a stream of `IMUWindow`s using fixed size + overlap
(a standard sliding-window segmentation, e.g. 2s windows at 50% overlap).

## Registry (`imu_features.core.registry`)

The registry is the plug-in mechanism. A feature is just a function:

```python
def my_feature(data, sample_rate) -> float: ...
```

registered with:

```python
from imu_features.core.registry import register_feature

@register_feature("my_feature", family="statistical", scope="channel",
                   min_samples=2, description="...")
def my_feature(x, sample_rate=None):
    return float(...)
```

- `scope="channel"` (default): `data` is a 1-D array. The engine calls this
  once per channel — x, y, z, and the resultant magnitude — for every
  sensor present in the window.
- `scope="triaxial"`: `data` is the full `(n_samples, 3)` array for one
  sensor. Use this when the feature genuinely needs all three axes at once
  (PCA shape descriptors, cross-axis correlation, orientation angles).
- `min_samples` lets a feature declare it needs a minimum window length
  (e.g. sample entropy needs enough points for reliable template matching);
  the engine returns `nan` instead of calling the function if the window is
  too short, rather than every feature re-implementing that guard.

Importing `imu_features.families` (which `imu_features/__init__.py` does
automatically) runs every family module's registration decorators as a
side effect, populating the global `REGISTRY`. Nothing about the engine
enumerates specific feature names — `REGISTRY.families()` and
`REGISTRY.all()` are the only ways it discovers what exists, so a new family
module dropped into `imu_features/families/` and imported from
`families/__init__.py` is immediately available with zero other changes.

## Engine (`imu_features.core.engine.FeatureEngine`)

`FeatureEngine(families=[...])` or `FeatureEngine(features=[...])` resolves
a subset of the registry once at construction time (fail fast on a typo'd
family/feature name). `engine.extract(window)` then produces a flat
`dict[str, float]`:

- Channel-scope features: `{sensor}_{channel}_{family}_{feature}`, e.g.
  `accel_x_statistical_mean`, `gyro_mag_frequency_dominant_frequency`.
- Triaxial-scope features: `{sensor}_triaxial_{family}_{feature}`, e.g.
  `accel_triaxial_geometrical_pca_linearity`.

This naming convention is deliberately flat and greppable — no nested
dicts — so the output plugs directly into a pandas DataFrame row
(`engine.extract_many(windows)` does this, falling back to a list of dicts
if pandas isn't installed) or any ML framework's feature vector.

## Why this shape

- **No god-object feature list.** Every family is a self-contained module
  with its own docstring explaining *why* those features exist and where
  they come from; `docs/FEATURE_TAXONOMY.md` is the connective research
  layer across all of them.
- **Orthogonal to sensor semantics.** The engine doesn't know or care what
  "accel" or "gyro" physically means — it just iterates over whatever
  sensors an `IMUWindow` happens to carry. A future `baro` or `emg` channel
  would work with zero engine changes as long as it's shaped `(n, 3)`
  (or the model is extended for scalar sensors — see "Extending further"
  below).
- **Selection over configuration.** There's no YAML/JSON feature config
  format to keep in sync with the code; `families=[...]` /
  `features=[...]` / `exclude=[...]` are plain Python and fail loudly on
  typos.

## Extending further

- **New feature, existing family**: add a function + `@register_feature`
  call to the family's module. Done.
- **New family**: create `imu_features/families/my_family.py` following the
  pattern in any existing family module, add `from . import my_family` to
  `imu_features/families/__init__.py`, and document the rationale in
  `docs/FEATURE_TAXONOMY.md`.
- **Scalar (non-triaxial) sensors** (e.g. a barometer or single-axis
  temperature channel): not yet modeled — `IMUWindow` currently requires
  shape `(n, 3)` per sensor. The natural extension is an optional
  `scalar_sensors: Dict[str, np.ndarray]` field on `IMUWindow` and a third
  registry scope (`scope="scalar"`) that the engine applies once per
  scalar sensor without synthesizing x/y/z/mag channels.
