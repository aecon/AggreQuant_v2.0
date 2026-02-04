"""
Data export utilities for statistics.

Exports results to CSV, Parquet, Excel, and text formats.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

import pandas as pd

from ..quantification.results import PlateResult, WellResult, FieldResult
from .controls import ControlComparison, get_control_statistics


def export_to_csv(
    data: Union[pd.DataFrame, PlateResult],
    output_path: Union[str, Path],
    level: str = "well",
) -> Path:
    """
    Export data to CSV format.

    Arguments:
        data: DataFrame or PlateResult to export
        output_path: Output file path
        level: "well" or "field" (only used if data is PlateResult)

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, PlateResult):
        df = data.to_dataframe(level=level)
    else:
        df = data

    df.to_csv(output_path, index=False)
    return output_path


def export_to_parquet(
    data: Union[pd.DataFrame, PlateResult],
    output_path: Union[str, Path],
    level: str = "well",
    compression: str = "snappy",
) -> Path:
    """
    Export data to Parquet format.

    Parquet is efficient for large datasets and preserves types.

    Arguments:
        data: DataFrame or PlateResult to export
        output_path: Output file path
        level: "well" or "field" (only used if data is PlateResult)
        compression: Compression algorithm ("snappy", "gzip", "brotli", None)

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, PlateResult):
        df = data.to_dataframe(level=level)
    else:
        df = data

    df.to_parquet(output_path, compression=compression, index=False)
    return output_path


def export_to_excel(
    data: Union[pd.DataFrame, PlateResult, Dict[str, pd.DataFrame]],
    output_path: Union[str, Path],
    sheet_name: str = "Results",
) -> Path:
    """
    Export data to Excel format.

    If data is a dict, each key becomes a sheet name.

    Arguments:
        data: DataFrame, PlateResult, or dict of DataFrames
        output_path: Output file path
        sheet_name: Sheet name (only used if data is single DataFrame)

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for name, df in data.items():
                df.to_excel(writer, sheet_name=name, index=False)
    elif isinstance(data, PlateResult):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Well-level data
            df_well = data.to_dataframe(level="well")
            df_well.to_excel(writer, sheet_name="Well_Results", index=False)

            # Field-level data
            df_field = data.to_dataframe(level="field")
            df_field.to_excel(writer, sheet_name="Field_Results", index=False)
    else:
        data.to_excel(output_path, sheet_name=sheet_name, index=False)

    return output_path


def export_plate_summary(
    plate_result: PlateResult,
    output_dir: Union[str, Path],
    prefix: str = "",
) -> Dict[str, Path]:
    """
    Export complete plate summary to multiple files.

    Creates:
    - {prefix}well_results.csv
    - {prefix}field_results.csv
    - {prefix}well_results.parquet
    - {prefix}control_statistics.txt
    - {prefix}summary.txt

    Arguments:
        plate_result: PlateResult to export
        output_dir: Output directory
        prefix: Optional prefix for filenames

    Returns:
        Dictionary mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Well-level CSV
    well_csv = output_dir / f"{prefix}well_results.csv"
    export_to_csv(plate_result, well_csv, level="well")
    files["well_csv"] = well_csv

    # Field-level CSV
    field_csv = output_dir / f"{prefix}field_results.csv"
    export_to_csv(plate_result, field_csv, level="field")
    files["field_csv"] = field_csv

    # Well-level Parquet
    well_parquet = output_dir / f"{prefix}well_results.parquet"
    export_to_parquet(plate_result, well_parquet, level="well")
    files["well_parquet"] = well_parquet

    # Summary text file
    summary_txt = output_dir / f"{prefix}summary.txt"
    _write_summary_text(plate_result, summary_txt)
    files["summary"] = summary_txt

    # Control statistics if controls are defined
    if plate_result.control_types:
        control_txt = output_dir / f"{prefix}control_statistics.txt"
        _write_control_statistics(plate_result, control_txt)
        files["control_stats"] = control_txt

    return files


def _write_summary_text(plate_result: PlateResult, output_path: Path):
    """Write plate summary to text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"AggreQuant Plate Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Plate Name: {plate_result.plate_name}\n")
        f.write(f"Plate Format: {plate_result.plate_format}-well\n")
        f.write(f"Timestamp: {plate_result.timestamp}\n\n")

        f.write("-" * 40 + "\n")
        f.write("Processing Summary\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total wells processed: {plate_result.total_n_wells_processed}\n")
        f.write(f"Total fields processed: {plate_result.total_n_fields_processed}\n")
        f.write(f"Total cells detected: {plate_result.total_n_cells}\n")
        f.write(f"Average cells per well: {plate_result.avg_cells_per_well:.1f}\n\n")

        if plate_result.ssmd is not None:
            f.write("-" * 40 + "\n")
            f.write("Quality Control\n")
            f.write("-" * 40 + "\n")
            f.write(f"SSMD: {plate_result.ssmd:.3f}\n")
            if plate_result.ssmd_control_pair:
                f.write(f"Controls: {plate_result.ssmd_control_pair[0]} vs {plate_result.ssmd_control_pair[1]}\n")
            f.write("\n")

        if plate_result.processing_time_seconds is not None:
            f.write(f"Processing time: {plate_result.processing_time_seconds:.1f} seconds\n")


def _write_control_statistics(plate_result: PlateResult, output_path: Path):
    """Write control well statistics to text file."""
    well_results = list(plate_result.well_results.values())
    stats = get_control_statistics(well_results)

    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Control Well Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Metric: Percentage of Aggregate-Positive Cells\n\n")

        # Header
        f.write(f"{'Control Type':<15} {'N':>5} {'Mean':>10} {'Std':>10} {'Median':>10}\n")
        f.write("-" * 55 + "\n")

        for control_type, stat in stats.items():
            f.write(
                f"{control_type:<15} "
                f"{stat['n']:>5} "
                f"{stat['mean']:>10.2f} "
                f"{stat['std']:>10.2f} "
                f"{stat['median']:>10.2f}\n"
            )

        f.write("\n")

        # Per-well values
        f.write("-" * 40 + "\n")
        f.write("Per-Well Values\n")
        f.write("-" * 40 + "\n\n")

        for control_type in stats:
            wells = [w for w in well_results if w.control_type == control_type]
            f.write(f"\n{control_type}:\n")
            for well in wells:
                f.write(f"  {well.well_id}: {well.pct_aggregate_positive_cells:.2f}%\n")


def export_density_map_data(
    plate_result: PlateResult,
    metric: str = "pct_aggregate_positive_cells",
) -> np.ndarray:
    """
    Create 2D array for plate density map visualization.

    Arguments:
        plate_result: PlateResult with well data
        metric: Metric to visualize

    Returns:
        2D numpy array with shape (n_rows, n_cols)
    """
    # Determine plate dimensions
    if plate_result.plate_format == "384":
        n_rows, n_cols = 16, 24
    else:
        n_rows, n_cols = 8, 12

    # Create grid
    grid = np.full((n_rows, n_cols), np.nan)

    for well in plate_result.well_results.values():
        # Parse well_id to get row and column
        row_idx = ord(well.row.upper()) - ord('A')
        col_idx = well.column - 1

        if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
            value = getattr(well, metric, None)
            if value is not None:
                grid[row_idx, col_idx] = value

    return grid


def create_qoi_table(
    field_results: List[FieldResult],
) -> pd.DataFrame:
    """
    Create QoI table in original AggreQuant format.

    Arguments:
        field_results: List of FieldResult objects

    Returns:
        DataFrame with QoI columns matching legacy format
    """
    data = []
    for f in field_results:
        data.append({
            "%Agg.Pos.Cells": f.pct_aggregate_positive_cells,
            "N.Cells": f.n_cells,
            "%Area.Agg.": f.pct_aggregate_area_over_cell,
            "AreaCells": f.total_cell_area_px,
            "%Ambig.Agg.": f.pct_ambiguous_aggregates,
            "N.Agg.Img(CC)": f.n_aggregates,
            "Avg.NAgg.perCell": f.avg_aggregates_per_positive_cell,
        })

    return pd.DataFrame(data)
