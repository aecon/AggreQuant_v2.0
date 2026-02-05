"""
Main processing pipeline orchestrator for AggreQuant.

Connects all components (image loading, segmentation, quantification,
statistics, export) into a complete analysis workflow.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Tuple
import numpy as np

from .common.logging import get_logger
from .loaders.config import PipelineConfig, ChannelConfig

logger = get_logger(__name__)
from .loaders.images import ImageLoader, group_files_by_field, load_tiff
from .loaders.plate import Plate, Well, FieldOfView, well_id_to_indices
from .quality.focus import compute_focus_metrics, generate_blur_mask
from .quantification.results import FieldResult, WellResult, PlateResult
from .quantification.measurements import (
    compute_field_measurements,
    compute_masked_measurements,
    apply_focus_metrics_to_result,
)
from .statistics.well_stats import aggregate_field_to_well
from .statistics.controls import compute_plate_ssmd, get_control_statistics
from .statistics.export import export_plate_summary


# Progress callback type
ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineState:
    """Tracks pipeline execution state."""
    is_running: bool = False
    is_cancelled: bool = False
    current_well: str = ""
    current_field: int = 0
    total_wells: int = 0
    total_fields: int = 0
    processed_wells: int = 0
    processed_fields: int = 0
    start_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)


class AggreQuantPipeline:
    """
    Main pipeline orchestrator for aggregate quantification.

    Coordinates image loading, segmentation, quantification, and export
    for complete plate analysis.

    Example usage:
        config = PipelineConfig.from_yaml("config.yaml")
        pipeline = AggreQuantPipeline(config)
        result = pipeline.run(progress_callback=my_callback)
    """

    def __init__(
        self,
        config: PipelineConfig,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the pipeline.

        Arguments:
            config: Pipeline configuration
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        self.config = config
        self.verbose = verbose or config.verbose
        self.debug = debug or config.debug

        # Pipeline state
        self.state = PipelineState()

        # Segmenters (lazy initialization)
        self._nuclei_segmenter = None
        self._cell_segmenter = None
        self._aggregate_segmenter = None

        # Results
        self.plate_result: Optional[PlateResult] = None

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            logger.info(message)

    def _debug(self, message: str):
        """Log debug message if debug mode enabled."""
        if self.debug:
            logger.debug(message)

    def _init_segmenters(self):
        """Initialize segmentation backends."""
        self._log("Initializing segmenters...")

        # Import here to allow lazy loading
        from .segmentation import (
            StarDistSegmenter,
            CellposeSegmenter,
            FilterBasedSegmenter,
            NeuralNetworkSegmenter,
        )

        seg_config = self.config.segmentation

        # Nuclei segmenter (StarDist)
        self._log(f"  Loading StarDist model: {seg_config.nuclei_model}")
        self._nuclei_segmenter = StarDistSegmenter(
            model_name=seg_config.nuclei_model,
            prob_thresh=seg_config.nuclei_prob_thresh,
            nms_thresh=seg_config.nuclei_nms_thresh,
        )

        # Cell segmenter (Cellpose)
        self._log(f"  Loading Cellpose model: {seg_config.cell_model}")
        self._cell_segmenter = CellposeSegmenter(
            model_type=seg_config.cell_model,
            diameter=seg_config.cell_diameter,
            flow_threshold=seg_config.cell_flow_threshold,
            cellprob_threshold=seg_config.cell_cellprob_threshold,
            use_gpu=self.config.use_gpu,
        )

        # Aggregate segmenter
        if seg_config.aggregate_method == "unet":
            self._log(f"  Loading UNet model: {seg_config.aggregate_model_path}")
            self._aggregate_segmenter = NeuralNetworkSegmenter(
                weights_path=seg_config.aggregate_model_path,
                min_aggregate_area=seg_config.aggregate_min_size,
                device="cuda" if self.config.use_gpu else "cpu",
            )
        else:
            self._log(f"  Using filter-based aggregate segmentation")
            self._aggregate_segmenter = FilterBasedSegmenter(
                normalized_threshold=seg_config.aggregate_intensity_threshold,
                min_aggregate_area=seg_config.aggregate_min_size,
            )

        self._log("Segmenters initialized.")

    def _get_channel_by_purpose(self, purpose: str) -> Optional[ChannelConfig]:
        """Get channel configuration by purpose."""
        for ch in self.config.channels:
            if ch.purpose == purpose:
                return ch
        return None

    def _discover_plate_structure(self) -> Tuple[ImageLoader, Dict[str, Dict[str, List[Path]]]]:
        """
        Discover plate structure from input directory.

        Returns:
            Tuple of (ImageLoader, dict mapping well_id -> field_id -> file_list)
        """
        # Build channel patterns from config
        channel_patterns = {}
        for ch in self.config.channels:
            channel_patterns[ch.name] = ch.pattern

        self._debug(f"Channel patterns: {channel_patterns}")

        # Create image loader
        loader = ImageLoader(
            directory=self.config.input_dir,
            channel_patterns=channel_patterns,
            verbose=self.verbose,
        )

        self._log(f"Discovered {loader.n_wells} wells")

        # Build field structure for each well
        plate_structure = {}
        for well_id in loader.wells:
            well_files = loader.get_well_files(well_id)
            fields = group_files_by_field(well_files)
            plate_structure[well_id] = fields

            if self.debug:
                self._debug(f"  {well_id}: {len(fields)} fields")

        return loader, plate_structure

    def _process_field(
        self,
        well_id: str,
        field_id: str,
        field_files: List[Path],
        plate_name: str,
    ) -> Optional[FieldResult]:
        """
        Process a single field of view.

        Arguments:
            well_id: Well identifier
            field_id: Field identifier
            field_files: List of image files for this field
            plate_name: Name of the plate

        Returns:
            FieldResult or None if processing failed
        """
        try:
            # Load images for each channel
            images = {}
            nuclei_ch = self._get_channel_by_purpose("nuclei")
            cell_ch = self._get_channel_by_purpose("cells")
            agg_ch = self._get_channel_by_purpose("aggregates")

            for ch in self.config.channels:
                matching_files = [f for f in field_files if ch.pattern.lower() in f.name.lower()]
                if matching_files:
                    images[ch.purpose] = load_tiff(matching_files[0])
                    self._debug(f"  Loaded {ch.purpose}: {matching_files[0].name}")

            if "nuclei" not in images:
                self._log(f"  Warning: No nuclei image found for {well_id}/{field_id}")
                return None

            if "cells" not in images:
                self._log(f"  Warning: No cell image found for {well_id}/{field_id}")
                return None

            if "aggregates" not in images:
                self._log(f"  Warning: No aggregate image found for {well_id}/{field_id}")
                return None

            # Segment nuclei
            self._debug(f"  Segmenting nuclei...")
            nuclei_labels = self._nuclei_segmenter.segment(images["nuclei"])
            self._debug(f"  Found {nuclei_labels.max()} nuclei")

            # Segment cells (using nuclei as seeds)
            self._debug(f"  Segmenting cells...")
            cell_labels = self._cell_segmenter.segment(
                images["cells"],
                nuclei_mask=nuclei_labels,
            )
            self._debug(f"  Found {cell_labels.max()} cells")

            # Segment aggregates
            self._debug(f"  Segmenting aggregates...")
            aggregate_labels = self._aggregate_segmenter.segment(images["aggregates"])
            self._debug(f"  Found {aggregate_labels.max()} aggregates")

            # Compute focus metrics on aggregate channel
            self._debug(f"  Computing focus metrics...")
            focus_metrics = compute_focus_metrics(
                images["aggregates"],
                patch_size=self.config.quality.focus_patch_size,
                blur_threshold=self.config.quality.focus_blur_threshold,
            )

            # Compute QoI measurements
            self._debug(f"  Computing measurements...")
            result, diagnostics = compute_field_measurements(
                cell_labels=cell_labels,
                aggregate_labels=aggregate_labels,
                nuclei_labels=nuclei_labels,
                verbose=self.verbose,
                debug=self.debug,
            )

            # Add focus metrics to result
            result = apply_focus_metrics_to_result(
                result,
                focus_metrics,
                blur_threshold=self.config.quality.focus_blur_threshold,
            )

            # Compute masked measurements if image has blur
            if focus_metrics.pct_patches_below_threshold > 0:
                blur_mask = generate_blur_mask(
                    images["aggregates"],
                    patch_size=self.config.quality.focus_patch_size,
                    blur_threshold=self.config.quality.focus_blur_threshold,
                )
                masked_stats = compute_masked_measurements(
                    cell_labels=cell_labels,
                    aggregate_labels=aggregate_labels,
                    blur_mask=blur_mask,
                )
                result.n_cells_masked = masked_stats["n_cells_masked"]
                result.n_aggregate_positive_cells_masked = masked_stats["n_aggregate_positive_cells_masked"]
                result.pct_aggregate_positive_cells_masked = masked_stats["pct_aggregate_positive_cells_masked"]
                result.total_cell_area_masked_px = masked_stats["total_cell_area_masked_px"]
                result.total_aggregate_area_masked_px = masked_stats["total_aggregate_area_masked_px"]

            # Fill in identifiers
            row, col = well_id_to_indices(well_id)
            result.plate_name = plate_name
            result.well_id = well_id
            result.row = chr(ord('A') + row)
            result.column = col + 1
            result.field = int(field_id)

            # Save masks if configured
            if self.config.output.save_masks:
                self._save_masks(well_id, field_id, nuclei_labels, cell_labels, aggregate_labels)

            return result

        except Exception as e:
            error_msg = f"Error processing {well_id}/{field_id}: {e}"
            self._log(error_msg)
            self.state.errors.append(error_msg)
            return None

    def _save_masks(
        self,
        well_id: str,
        field_id: str,
        nuclei_labels: np.ndarray,
        cell_labels: np.ndarray,
        aggregate_labels: np.ndarray,
    ):
        """Save segmentation masks to output directory."""
        import tifffile

        mask_dir = self.config.output.output_dir / "masks" / well_id
        mask_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(mask_dir / f"f{field_id}_nuclei.tif", nuclei_labels)
        tifffile.imwrite(mask_dir / f"f{field_id}_cells.tif", cell_labels)
        tifffile.imwrite(mask_dir / f"f{field_id}_aggregates.tif", aggregate_labels)

    def _process_well(
        self,
        well_id: str,
        field_structure: Dict[str, List[Path]],
        plate_name: str,
        control_type: Optional[str],
    ) -> Optional[WellResult]:
        """
        Process all fields in a well.

        Arguments:
            well_id: Well identifier
            field_structure: Dict mapping field_id to file list
            plate_name: Name of the plate
            control_type: Control type assignment (or None)

        Returns:
            WellResult or None if no fields were processed
        """
        field_results = []

        for field_id, field_files in sorted(field_structure.items()):
            if self.state.is_cancelled:
                break

            self.state.current_field = int(field_id) if field_id.isdigit() else 0

            result = self._process_field(well_id, field_id, field_files, plate_name)
            if result is not None:
                field_results.append(result)

            self.state.processed_fields += 1

        if not field_results:
            return None

        # Aggregate to well level
        row, col = well_id_to_indices(well_id)
        well_result = aggregate_field_to_well(
            field_results=field_results,
            plate_name=plate_name,
            well_id=well_id,
            row=chr(ord('A') + row),
            column=col + 1,
            control_type=control_type,
        )

        return well_result

    def _get_control_type(self, well_id: str) -> Optional[str]:
        """Get control type for a well from config."""
        for control_type, wells in self.config.control_wells.items():
            if well_id in wells:
                return control_type
        return None

    def run(
        self,
        progress_callback: Optional[ProgressCallback] = None,
        wells_to_process: Optional[List[str]] = None,
    ) -> PlateResult:
        """
        Run the complete analysis pipeline.

        Arguments:
            progress_callback: Optional callback(progress: float, message: str)
            wells_to_process: Optional list of specific wells to process

        Returns:
            PlateResult with all analysis results
        """
        self._log("Starting AggreQuant pipeline...")

        self.state = PipelineState(is_running=True, start_time=time.time())

        def report_progress(progress: float, message: str):
            if progress_callback:
                progress_callback(progress, message)
            self._log(message)

        try:
            # Discover plate structure
            report_progress(0.0, "Discovering plate structure...")
            loader, plate_structure = self._discover_plate_structure()

            # Filter wells if specified
            if wells_to_process:
                plate_structure = {w: f for w, f in plate_structure.items() if w in wells_to_process}

            # Count total fields
            self.state.total_wells = len(plate_structure)
            self.state.total_fields = sum(len(fields) for fields in plate_structure.values())

            self._log(f"Processing {self.state.total_wells} wells, {self.state.total_fields} fields")

            # Initialize segmenters
            report_progress(0.05, "Loading segmentation models...")
            self._init_segmenters()

            # Create output directory
            self.config.output.output_dir.mkdir(parents=True, exist_ok=True)

            # Process wells
            plate_name = self.config.input_dir.name
            well_results = {}

            for i, (well_id, field_structure) in enumerate(sorted(plate_structure.items())):
                if self.state.is_cancelled:
                    report_progress(0.0, "Pipeline cancelled by user")
                    break

                self.state.current_well = well_id
                progress = 0.1 + 0.8 * (i / self.state.total_wells)
                report_progress(progress, f"Processing well {well_id}...")

                control_type = self._get_control_type(well_id)
                well_result = self._process_well(
                    well_id, field_structure, plate_name, control_type
                )

                if well_result is not None:
                    well_results[well_id] = well_result

                self.state.processed_wells += 1

            # Build plate result
            report_progress(0.9, "Computing plate statistics...")

            # Determine control types present
            control_types = set()
            for well_result in well_results.values():
                if well_result.control_type:
                    control_types.add(well_result.control_type)

            # Create plate result
            self.plate_result = PlateResult(
                plate_name=plate_name,
                plate_format=self.config.plate_format,
                well_results=well_results,
                control_types=control_types,
                timestamp=datetime.now().isoformat(),
                processing_time_seconds=time.time() - self.state.start_time,
            )

            # Compute SSMD if controls are defined
            if "negative" in control_types and len(control_types) > 1:
                # Find positive control type
                pos_type = None
                for ct in control_types:
                    if ct != "negative":
                        pos_type = ct
                        break

                if pos_type:
                    ssmd = compute_plate_ssmd(
                        list(well_results.values()),
                        positive_control=pos_type,
                        negative_control="negative",
                    )
                    self.plate_result.ssmd = ssmd
                    self.plate_result.ssmd_control_pair = (pos_type, "negative")

            # Export results
            if self.config.output.save_statistics:
                report_progress(0.95, "Exporting results...")
                self._export_results()

            report_progress(1.0, "Pipeline complete!")
            self._log(f"Processed {self.state.processed_wells} wells, {self.state.processed_fields} fields")

            if self.state.errors:
                self._log(f"Encountered {len(self.state.errors)} errors")

            return self.plate_result

        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            self._log(error_msg)
            self.state.errors.append(error_msg)
            raise

        finally:
            self.state.is_running = False

    def _export_results(self):
        """Export plate results to files."""
        if self.plate_result is None:
            return

        output_dir = self.config.output.output_dir
        files = export_plate_summary(
            self.plate_result,
            output_dir,
            prefix=f"{self.plate_result.plate_name}_",
        )

        self._log(f"Exported results to {output_dir}:")
        for name, path in files.items():
            self._log(f"  {name}: {path}")

    def cancel(self):
        """Request cancellation of the running pipeline."""
        self.state.is_cancelled = True
        self._log("Cancellation requested")

    def get_state(self) -> PipelineState:
        """Get current pipeline execution state."""
        return self.state


def run_pipeline_from_config(
    config_path: Path,
    progress_callback: Optional[ProgressCallback] = None,
    verbose: bool = True,
) -> PlateResult:
    """
    Convenience function to run pipeline from a YAML config file.

    Arguments:
        config_path: Path to YAML configuration file
        progress_callback: Optional progress callback
        verbose: Print progress messages

    Returns:
        PlateResult with all analysis results
    """
    config = PipelineConfig.from_yaml(config_path)
    pipeline = AggreQuantPipeline(config, verbose=verbose)
    return pipeline.run(progress_callback=progress_callback)


def run_pipeline_from_dict(
    config_dict: Dict[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    verbose: bool = True,
) -> PlateResult:
    """
    Run pipeline from a configuration dictionary (e.g., from GUI).

    Arguments:
        config_dict: Configuration as dictionary
        progress_callback: Optional progress callback
        verbose: Print progress messages

    Returns:
        PlateResult with all analysis results
    """
    # Build channel configs
    channels = []
    for ch_data in config_dict.get("channels", []):
        channels.append(ChannelConfig(**ch_data))

    # If no channels specified, use defaults
    if not channels:
        channels = [
            ChannelConfig(name="DAPI", pattern="C01", purpose="nuclei"),
            ChannelConfig(name="GFP", pattern="C02", purpose="aggregates"),
            ChannelConfig(name="CellMask", pattern="C03", purpose="cells"),
        ]

    # Build config
    from .loaders.config import SegmentationConfig, QualityConfig, OutputConfig

    seg_config = SegmentationConfig(
        aggregate_method=config_dict.get("aggregate_method", "unet"),
        aggregate_model_path=config_dict.get("model_path"),
    )

    quality_config = QualityConfig(
        focus_blur_threshold=config_dict.get("blur_threshold", 15.0),
        focus_reject_threshold=config_dict.get("blur_reject_pct", 50.0),
    )

    output_config = OutputConfig(
        output_dir=Path(config_dict.get("output_dir", "output")),
        save_masks=config_dict.get("save_masks", True),
        save_overlays=config_dict.get("save_overlays", True),
        save_statistics=True,
    )

    # Convert control_wells from {well: type} to {type: [wells]}
    control_wells_input = config_dict.get("control_wells", {})
    control_wells = {}
    for well, ctrl_type in control_wells_input.items():
        if ctrl_type not in control_wells:
            control_wells[ctrl_type] = []
        control_wells[ctrl_type].append(well)

    config = PipelineConfig(
        input_dir=Path(config_dict.get("input_dir", ".")),
        plate_format=config_dict.get("plate_format", "96"),
        channels=channels,
        segmentation=seg_config,
        quality=quality_config,
        output=output_config,
        control_wells=control_wells,
        use_gpu=config_dict.get("use_gpu", True),
        verbose=verbose,
    )

    pipeline = AggreQuantPipeline(config, verbose=verbose)
    return pipeline.run(progress_callback=progress_callback)
