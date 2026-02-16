# AggreQuant — Publication Progress

Tracking progress toward a working pipeline and its companion manuscript.

## Phase 1: Get the segmentation pipeline running

- [x] Fix `run_pipeline.py` to use `SegmentationPipeline` correctly
- [x] Run pipeline on plate data (nuclei + cells + aggregates via filter method)
- [x] Clean up pipeline internals (vectorize relabeling, fix square-image bug, remove `load_tiff`)
- [x] Add configurable per-image focus quality metrics (patch-based + global, saved to CSV)
- [x] Verify output masks are correct (visual inspection)

## Phase 2: Quantification and statistics

- [x] Per-field measurements (n_cells, nuclei/cell/aggregate areas, % agg-positive cells → CSV)
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
