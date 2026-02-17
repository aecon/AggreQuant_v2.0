# AggreQuant — Progress Overview

Tracking progress toward a working pipeline and its companion manuscript.

## Phase 1: Segmentation pipeline

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

## Phase 3: Code quality and refactoring

- [x] Extract post-processing into reusable module (`segmentation/postprocessing.py`)
- [x] Unify logging — replace `print`/`_log` with `get_logger` throughout pipeline
- [x] Use `FieldResult` end-to-end instead of manual dict flattening
- [x] Move TensorFlow GPU config to `common/gpu_utils.py` (idempotent)
- [x] Selective patch focus metrics (only compute requested metrics)
- [x] Delete dead code (`pipeline.bak.py`)
- [x] Use absolute imports throughout the package
- [x] Clean file headers (one-liner docstrings, remove author lines)
- [ ] Wire `aggregate_method` config to actual segmenter selection
- [ ] Fix dead code path in `image_utils.py` (unreachable tifffile fallback)
- [ ] Bridge `Plate`/`Well` structures into the pipeline

## Phase 4: Neural network aggregate segmentation

- [ ] Train UNet on annotated aggregate data
- [ ] Benchmark UNet vs filter-based segmentation
- [ ] Select best architecture variant

## Phase 5: Manuscript

- [ ] Introduction and related work
- [ ] Methods: pipeline architecture, focus metrics, segmentation
- [ ] Results: benchmarks, plate-level statistics
- [ ] Discussion
- [ ] Figures

---

## Changelog

**2026-02-17**
- Extract post-processing into reusable module (`segmentation/postprocessing.py`)
- Unify logging — replace `print`/`_log` with `get_logger` throughout pipeline
- Use `FieldResult` end-to-end instead of manual dict flattening
- Move TensorFlow GPU config to `common/gpu_utils.py` (idempotent)
- Selective patch focus metrics (only compute requested metrics)
- Delete dead code (`pipeline.bak.py`)
- Use absolute imports throughout the package
- Clean file headers (one-liner docstrings, remove author lines)
