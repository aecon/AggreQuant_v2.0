"""Configuration presets for benchmark architectures.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .presets import (
    BENCHMARK_CONFIGS,
    get_config,
    list_configs,
    get_config_description,
    print_configs,
)

__all__ = [
    "BENCHMARK_CONFIGS",
    "get_config",
    "list_configs",
    "get_config_description",
    "print_configs",
]
