"""Open surrogate for the unpublished MLX architecture simulator."""

from .schema import CalibrationConfig, HardwareConfig, SimulationResult, Workload
from .simulator import MLXSimulator

__all__ = [
    "CalibrationConfig",
    "HardwareConfig",
    "MLXSimulator",
    "SimulationResult",
    "Workload",
]
__version__ = "0.1.0"
