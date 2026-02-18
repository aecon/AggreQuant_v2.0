# AggreQuant — Progress Overview

Tracking progress toward a working pipeline and its companion manuscript.

---

## Milestones

### Phase 1: Image segmentation pipeline — *complete*

- [x] Multi-channel image loading and plate layout parsing
- [x] Nuclei, cell, and aggregate segmentation (classical methods)
- [x] Focus quality assessment
- [x] Per-field quantification

### Phase 2: Plate-level statistics and export

- [ ] Well-level aggregation of field measurements
- [ ] Assay quality control (SSMD, Z-factor)
- [ ] Structured data export

### Phase 3: Deep learning aggregate segmentation

- [ ] UNet training on annotated data
- [ ] Benchmarking against classical segmentation

### Phase 4: Manuscript

- [ ] Methods and results write-up
- [ ] Figures and benchmarks

---

## Future improvements

- Dead code removal
- Over-engineered abstractions
- Incomplete pipeline wiring
- Missing tests and validation
- Redundant/duplicate code
- Documentation staleness
- Remaining quick fixes
- Packaging and separation of concerns

---

## Changelog

**2026-02-17** ([Plan](Plan_2026-02-17.md))
- Optimize `compute_field_measurements` with sparse cross-tabulation (~6x speedup)
- Extract post-processing into reusable module (`segmentation/postprocessing.py`)
- Unify logging — replace `print`/`_log` with `get_logger` throughout pipeline
- Use `FieldResult` end-to-end instead of manual dict flattening
- Move TensorFlow GPU config to `common/gpu_utils.py` (idempotent)
- Selective patch focus metrics (only compute requested metrics)
- Delete dead code (`pipeline.bak.py`)
- Use absolute imports throughout the package
- Clean file headers (one-liner docstrings, remove author lines)
- Merge focus and field CSVs into single output, use relative `output_subdir`
- Refactor quantification: new `colocalization.py` module, plain dicts, delete `FieldResult`
- Flatten `quality/` and `quantification/` packages into top-level modules
- Define `ALL_PATCH_METRICS` in one step (remove empty-dict-then-update pattern)
- Extract `compute_focus_metrics` from pipeline into `focus.py` as public API
- Remove duplicated metric validation sets from `config.py` (import from `focus.py`)
- Replace `_normalize_to_8bit` with `image_utils.normalize_image`
- Move `remove_small_*_compat` from `image_utils` to `segmentation/postprocessing.py`
- Remove Operetta/ImageXpress parsers, keep only InCell format
- Add `plate_name` config field and InCell parser tests
