"""Configuration schema and validation for the analysis pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml


@dataclass
class ChannelConfig:
    """Configuration for a single imaging channel."""
    name: str
    pattern: str  # File pattern (e.g., "C01" for DAPI)
    purpose: str  # "nuclei", "cells", "aggregates", "other"
    normalize_method: str = "percentile"

    def __post_init__(self):
        valid_purposes = {"nuclei", "cells", "aggregates", "other"}
        if self.purpose not in valid_purposes:
            raise ValueError(f"purpose must be one of {valid_purposes}, got '{self.purpose}'")


@dataclass
class SegmentationConfig:
    """Configuration for segmentation parameters."""
    # Nuclei segmentation (StarDist)
    nuclei_sigma_denoise: float = 2.0
    nuclei_sigma_background: float = 50.0
    nuclei_min_area: int = 300
    nuclei_max_area: int = 15000

    # Cell segmentation (Cellpose)
    cell_model: str = "cyto3"

    # Aggregate segmentation
    aggregate_method: str = "unet"  # "unet", "filter", "hybrid"
    aggregate_model_path: Optional[Path] = None
    aggregate_min_size: int = 10  # pixels
    aggregate_intensity_threshold: float = 0.5


VALID_PATCH_METRICS = {"VarianceLaplacian", "LaplaceEnergy", "Sobel", "Brenner", "FocusScore"}
VALID_GLOBAL_METRICS = {"power_log_log_slope", "global_variance_laplacian", "high_freq_ratio"}
VALID_FOCUS_CHANNELS = {"nuclei", "cells"}


@dataclass
class QualityConfig:
    """Configuration for focus quality metrics."""
    # Which channels to compute focus metrics on (empty = skip focus entirely)
    compute_on: List[str] = field(default_factory=list)

    # Which categories to compute
    compute_patch_metrics: bool = True
    compute_global_metrics: bool = True

    # Which specific metrics to compute
    patch_metrics: List[str] = field(default_factory=lambda: ["VarianceLaplacian"])
    global_metrics: List[str] = field(default_factory=lambda: ["power_log_log_slope"])

    # Patch settings
    patch_size: Tuple[int, int] = (40, 40)

    def __post_init__(self):
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)

        for ch in self.compute_on:
            if ch not in VALID_FOCUS_CHANNELS:
                raise ValueError(
                    f"Invalid focus channel '{ch}'. Must be one of {VALID_FOCUS_CHANNELS}"
                )
        for m in self.patch_metrics:
            if m not in VALID_PATCH_METRICS:
                raise ValueError(
                    f"Invalid patch metric '{m}'. Must be one of {VALID_PATCH_METRICS}"
                )
        for m in self.global_metrics:
            if m not in VALID_GLOBAL_METRICS:
                raise ValueError(
                    f"Invalid global metric '{m}'. Must be one of {VALID_GLOBAL_METRICS}"
                )


@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_subdir: str = "aggrequant_output"
    save_masks: bool = True
    save_overlays: bool = True
    save_statistics: bool = True
    statistics_format: str = "parquet"  # "parquet", "csv", "both"


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    This is the main configuration object that ties together all settings
    for the analysis pipeline.
    """
    # Required fields
    input_dir: Path
    plate_format: str  # "96" or "384"

    # Channel configuration
    channels: List[ChannelConfig] = field(default_factory=list)

    # Segmentation settings
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    # Quality control
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Output settings
    output: OutputConfig = field(default_factory=OutputConfig)

    # Control wells (for normalization)
    control_wells: Dict[str, List[str]] = field(default_factory=dict)
    # Example: {"negative": ["A01", "A02"], "positive": ["H11", "H12"]}

    # Processing options
    n_workers: int = 4
    use_gpu: bool = True
    verbose: bool = False
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)

        valid_formats = {"96", "384"}
        if self.plate_format not in valid_formats:
            raise ValueError(f"plate_format must be one of {valid_formats}")

    @property
    def output_dir(self) -> Path:
        """Resolve output directory as input_dir / output.output_subdir."""
        return self.input_dir / self.output.output_subdir

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """
        Load configuration from YAML file.

        Arguments:
            path: Path to YAML config file

        Returns:
            PipelineConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse channels
        channels = []
        for ch_data in data.pop("channels", []):
            channels.append(ChannelConfig(**ch_data))

        # Parse nested configs
        seg_data = data.pop("segmentation", {})
        segmentation = SegmentationConfig(**seg_data)

        quality_data = data.pop("quality", {})
        quality = QualityConfig(**quality_data)

        output_data = data.pop("output", {})
        output = OutputConfig(**output_data)

        return cls(
            channels=channels,
            segmentation=segmentation,
            quality=quality,
            output=output,
            **data
        )

    def to_yaml(self, path: Path) -> None:
        """
        Save configuration to YAML file.

        Arguments:
            path: Path to save YAML config
        """
        data = {
            "input_dir": str(self.input_dir),
            "plate_format": self.plate_format,
            "channels": [
                {"name": ch.name, "pattern": ch.pattern, "purpose": ch.purpose,
                 "normalize_method": ch.normalize_method}
                for ch in self.channels
            ],
            "segmentation": {
                "nuclei_sigma_denoise": self.segmentation.nuclei_sigma_denoise,
                "nuclei_sigma_background": self.segmentation.nuclei_sigma_background,
                "nuclei_min_area": self.segmentation.nuclei_min_area,
                "nuclei_max_area": self.segmentation.nuclei_max_area,
                "cell_model": self.segmentation.cell_model,
                "aggregate_method": self.segmentation.aggregate_method,
                "aggregate_model_path": str(self.segmentation.aggregate_model_path) if self.segmentation.aggregate_model_path else None,
                "aggregate_min_size": self.segmentation.aggregate_min_size,
                "aggregate_intensity_threshold": self.segmentation.aggregate_intensity_threshold,
            },
            "quality": {
                "compute_on": self.quality.compute_on,
                "compute_patch_metrics": self.quality.compute_patch_metrics,
                "compute_global_metrics": self.quality.compute_global_metrics,
                "patch_metrics": self.quality.patch_metrics,
                "global_metrics": self.quality.global_metrics,
                "patch_size": list(self.quality.patch_size),
            },
            "output": {
                "output_subdir": self.output.output_subdir,
                "save_masks": self.output.save_masks,
                "save_overlays": self.output.save_overlays,
                "save_statistics": self.output.save_statistics,
                "statistics_format": self.output.statistics_format,
            },
            "control_wells": self.control_wells,
            "n_workers": self.n_workers,
            "use_gpu": self.use_gpu,
            "verbose": self.verbose,
            "debug": self.debug,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_default_config(
    input_dir: Path,
    plate_format: str = "96"
) -> PipelineConfig:
    """
    Create a default configuration for typical HCS analysis.

    Arguments:
        input_dir: Path to input images
        plate_format: "96" or "384" well plate

    Returns:
        PipelineConfig with sensible defaults
    """
    return PipelineConfig(
        input_dir=input_dir,
        plate_format=plate_format,
        channels=[
            ChannelConfig(name="DAPI", pattern="C01", purpose="nuclei"),
            ChannelConfig(name="GFP", pattern="C02", purpose="aggregates"),
            ChannelConfig(name="CellMask", pattern="C03", purpose="cells"),
        ],
    )
