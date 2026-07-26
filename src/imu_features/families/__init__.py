"""Importing this module registers every built-in feature family with the
global registry (imu_features imports it automatically on package import)."""

from . import coupling  # noqa: F401
from . import crossaxis  # noqa: F401
from . import frequency  # noqa: F401
from . import gait  # noqa: F401
from . import geometrical  # noqa: F401
from . import magnitude  # noqa: F401
from . import mechanical  # noqa: F401
from . import nonlinear  # noqa: F401
from . import orientation  # noqa: F401
from . import statistical  # noqa: F401
from . import topological  # noqa: F401
from . import wavelet  # noqa: F401
