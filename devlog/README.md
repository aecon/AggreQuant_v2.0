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

**2026-02-17**
- Extract post-processing into reusable module (`segmentation/postprocessing.py`)
- Unify logging — replace `print`/`_log` with `get_logger` throughout pipeline
- Use `FieldResult` end-to-end instead of manual dict flattening
- Move TensorFlow GPU config to `common/gpu_utils.py` (idempotent)
- Selective patch focus metrics (only compute requested metrics)
- Delete dead code (`pipeline.bak.py`)
- Use absolute imports throughout the package
- Clean file headers (one-liner docstrings, remove author lines)
