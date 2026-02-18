import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def print_header(text: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def print_step(text: str):
    """Print a step indicator."""
    print(f"\n>>> {text}")


def main():
    print_header("AggreQuant Minimal Pipeline Test")

    # =========================================================================
    # Step 1: Load sample images
    # =========================================================================
    print_step("Step 1: Loading sample images")

    from aggrequant.loaders.images import load_tiff

    data_dir = project_root / "tests" / "data"

    # Sample files
    nuclei_file = data_dir / "Plate_HA13rep1_K - 13(fld 3 wv 390 - Blue).tif"
    aggregate_file = data_dir / "Plate_HA13rep1_K - 13(fld 3 wv 473 - Green2).tif"
    cell_file = data_dir / "Plate_HA13rep1_K - 13(fld 3 wv 631 - FarRed).tif"

    # Load images
    nuclei_img = load_tiff(nuclei_file)
    aggregate_img = load_tiff(aggregate_file)
    cell_img = load_tiff(cell_file)

    print(f"  Nuclei image:     shape={nuclei_img.shape}, dtype={nuclei_img.dtype}")
    print(f"  Aggregate image:  shape={aggregate_img.shape}, dtype={aggregate_img.dtype}")
    print(f"  Cell image:       shape={cell_img.shape}, dtype={cell_img.dtype}")
    print(f"  ✓ Images loaded successfully")

    # =========================================================================
    # Step 2: Focus quality assessment
    # =========================================================================
    print_step("Step 2: Focus quality assessment")

    from aggrequant.quality.focus import compute_focus_metrics, generate_blur_mask

    # Compute focus metrics for nuclei channel (most reliable for focus)
    focus_metrics = compute_focus_metrics(
        nuclei_img,
        patch_size=(40, 40),
        blur_threshold=15.0
    )

    print(f"  Mean variance of Laplacian: {focus_metrics.variance_laplacian_mean:.2f}")
    print(f"  Patches below threshold:    {focus_metrics.pct_patches_blurry:.1f}%")
    print(f"  Is likely blurry:           {focus_metrics.is_likely_blurry}")

    # Generate blur mask
    blur_mask = generate_blur_mask(nuclei_img, patch_size=(40, 40), blur_threshold=15.0)
    pct_valid = (1 - blur_mask.mean()) * 100
    print(f"  Valid (non-blurry) area:    {pct_valid:.1f}%")
    print(f"  ✓ Focus assessment complete")

    # =========================================================================
    # Step 3: Nuclei segmentation
    # =========================================================================
    print_step("Step 3: Nuclei segmentation (StarDist)")

    try:
        from aggrequant.segmentation.stardist import StarDistSegmenter

        nuclei_segmenter = StarDistSegmenter(
            model_name="2D_versatile_fluo",
            verbose=False
        )

        nuclei_labels = nuclei_segmenter.segment(nuclei_img)
        n_nuclei = nuclei_labels.max()
        print(f"  Detected nuclei: {n_nuclei}")
        print(f"  ✓ Nuclei segmentation complete")

    except ImportError as e:
        print(f"  ⚠ StarDist not available: {e}")
        print(f"  Creating mock nuclei labels for testing...")
        # Create simple mock segmentation using thresholding
        from skimage import filters, morphology, measure
        threshold = filters.threshold_otsu(nuclei_img)
        binary = nuclei_img > threshold
        binary = morphology.remove_small_objects(binary, min_size=100)
        nuclei_labels = measure.label(binary)
        n_nuclei = nuclei_labels.max()
        print(f"  Mock nuclei (Otsu threshold): {n_nuclei}")

    # =========================================================================
    # Step 4: Cell segmentation
    # =========================================================================
    print_step("Step 4: Cell segmentation (Cellpose)")

    try:
        from aggrequant.segmentation.cellpose import CellposeSegmenter

        cell_segmenter = CellposeSegmenter(
            model_type="cyto2",
            verbose=False
        )

        cell_labels = cell_segmenter.segment(cell_img, nuclei_labels=nuclei_labels)
        n_cells = cell_labels.max()
        print(f"  Detected cells: {n_cells}")
        print(f"  ✓ Cell segmentation complete")

    except ImportError as e:
        print(f"  ⚠ Cellpose not available: {e}")
        print(f"  Using nuclei labels as cell approximation...")
        cell_labels = nuclei_labels.copy()
        n_cells = cell_labels.max()
        print(f"  Cells (from nuclei): {n_cells}")

    # =========================================================================
    # Step 5: Aggregate segmentation (filter-based)
    # =========================================================================
    print_step("Step 5: Aggregate segmentation (filter-based)")

    from aggrequant.segmentation.aggregates import FilterBasedSegmenter

    agg_segmenter = FilterBasedSegmenter(
        normalized_threshold=1.6,
        min_aggregate_area=9,
        verbose=False
    )

    aggregate_labels = agg_segmenter.segment(aggregate_img)
    n_aggregates = aggregate_labels.max()
    print(f"  Detected aggregates: {n_aggregates}")
    print(f"  ✓ Aggregate segmentation complete")

    # =========================================================================
    # Step 6: Quantification
    # =========================================================================
    print_step("Step 6: Computing quantification metrics")

    from aggrequant.quantification.measurements import compute_field_measurements

    # Compute measurements - returns (FieldResult, diagnostics dict)
    field_result, diagnostics = compute_field_measurements(
        cell_labels=cell_labels,
        aggregate_labels=aggregate_labels,
        nuclei_labels=nuclei_labels,
    )

    print(f"  Total cells:                    {field_result.n_cells}")
    print(f"  Aggregate-positive cells:       {field_result.n_aggregate_positive_cells}")
    print(f"  Percentage aggregate-positive:  {field_result.pct_aggregate_positive_cells:.2f}%")
    print(f"  Total aggregates:               {field_result.n_aggregates}")
    print(f"  ✓ Quantification complete")

    # =========================================================================
    # Step 7: Update field result with metadata
    # =========================================================================
    print_step("Step 7: Updating result with metadata")

    # Update field result with identifiers and focus metrics
    field_result.plate_name = "Plate_HA13rep1"
    field_result.well_id = "K13"
    field_result.row = "K"
    field_result.column = 13
    field_result.field = 3
    field_result.focus_variance_laplacian_mean = focus_metrics.variance_laplacian_mean
    field_result.focus_pct_patches_blurry = focus_metrics.pct_patches_blurry
    field_result.focus_is_likely_blurry = focus_metrics.is_likely_blurry
    field_result.segmentation_method = "filter"

    print(f"  FieldResult updated for K13, field 3")

    # Create well result
    from aggrequant.statistics.well_stats import aggregate_field_to_well

    well_result = aggregate_field_to_well(
        field_results=[field_result],
        plate_name="Plate_HA13rep1",
        well_id="K13",
        row="K",
        column=13,
    )
    print(f"  WellResult created for K13")
    print(f"    - Fields processed: {well_result.n_fields}")
    print(f"    - Total cells: {well_result.total_n_cells}")
    print(f"    - Pct aggregate-positive: {well_result.pct_aggregate_positive_cells:.2f}%")

    # Create plate result
    from aggrequant.quantification.results import PlateResult

    plate_result = PlateResult(
        plate_name="Plate_HA13rep1",
        plate_format="96",
        well_results={"K13": well_result},
        total_n_wells_processed=1,
        total_n_fields_processed=1,
        total_n_cells=well_result.total_n_cells,
        avg_cells_per_well=float(well_result.total_n_cells),
    )

    print(f"  PlateResult created for Plate_HA13rep1")
    print(f"  ✓ Result objects created")

    # =========================================================================
    # Step 8: Export results
    # =========================================================================
    print_step("Step 8: Exporting results")

    from aggrequant.statistics.export import export_plate_summary

    output_dir = project_root / "tests" / "data" / "output"
    output_dir.mkdir(exist_ok=True)

    # Export summary (creates multiple files in output_dir)
    exported_files = export_plate_summary(plate_result, output_dir, prefix="test_")
    for file_type, file_path in exported_files.items():
        print(f"  Exported {file_type}: {file_path.name}")

    print(f"  ✓ Export complete")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Test Summary")

    print(f"""
    Input:  Well K13, Field 3
    Images: {nuclei_img.shape[0]}x{nuclei_img.shape[1]} pixels

    Results:
      - Nuclei detected:        {n_nuclei}
      - Cells detected:         {n_cells}
      - Aggregates detected:    {n_aggregates}
      - Aggregate-positive:     {field_result.pct_aggregate_positive_cells:.2f}%
      - Focus score:            {focus_metrics.variance_laplacian_mean:.2f}
      - Image quality:          {'GOOD' if not focus_metrics.is_likely_blurry else 'BLURRY'}

    Output: {output_dir}
    """)

    print("="*60)
    print("  ✓ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
