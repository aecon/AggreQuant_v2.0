# AggreQuant — Publication Progress

Tracking progress toward a working pipeline and its companion manuscript.

## Phase 1: Get the segmentation pipeline running

- [ ] Fix `run_pipeline.py` to use `SegmentationPipeline` correctly
- [ ] Run pipeline on plate data (nuclei + cells + aggregates via filter method)
- [ ] Verify output masks are correct (visual inspection)
- [ ] Ensure focus quality check is integrated

## Phase 2: Quantification and statistics

- [ ] Per-cell measurements (aggregate count, area, intensity)
- [ ] Well-level statistics (% aggregate-positive cells)
- [ ] Control well analysis (SSMD, Z-factor)
- [ ] Export results (CSV / Parquet)

## Phase 3: Neural network aggregate segmentation

- [ ] Train UNet on annotated aggregate data
- [ ] Benchmark UNet vs filter-based segmentation
- [ ] Select best architecture variant

## Phase 4: Manuscript

- [ ] Introduction and related work
- [ ] Methods: pipeline architecture, focus metrics, segmentation
- [ ] Results: benchmarks, plate-level statistics
- [ ] Discussion
- [ ] Figures
