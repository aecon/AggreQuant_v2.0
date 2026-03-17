"""Background pipeline runner for the web GUI.

Runs SegmentationPipeline in a background thread and exposes progress
via a shared state dict that Dash callbacks can poll.
"""

import threading
import traceback
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PlateJob:
    """A single plate to process."""
    input_dir: str
    plate_name: str = ""
    config_path: Optional[str] = None  # per-plate YAML override
    status: str = "pending"  # pending | running | done | failed | cancelled
    error: str = ""


@dataclass
class RunnerState:
    """Shared state between the runner thread and Dash callbacks."""
    # Batch queue
    jobs: List[PlateJob] = field(default_factory=list)

    # Overall progress
    current_job_index: int = 0
    total_jobs: int = 0

    # Per-plate progress
    current_field: int = 0
    total_fields: int = 0
    current_status: str = "idle"  # idle | running | done | cancelled | failed

    # Log messages: list of (timestamp, level, message)
    log: List[tuple] = field(default_factory=list)

    # Control flags
    cancel_requested: bool = False
    is_running: bool = False

    def reset(self):
        self.current_job_index = 0
        self.total_jobs = 0
        self.current_field = 0
        self.total_fields = 0
        self.current_status = "idle"
        self.log.clear()
        self.cancel_requested = False
        self.is_running = False

    def add_log(self, level, message):
        ts = time.strftime("%H:%M:%S")
        self.log.append((ts, level, message))


# Global shared state (single-user app)
state = RunnerState()


def _build_config_from_gui(gui_config: dict, plate_job: PlateJob) -> Path:
    """Write a temporary YAML config for the pipeline.

    Takes the GUI settings dict and a PlateJob, writes a YAML file
    to a temp location, and returns the path.
    """
    import tempfile
    import yaml

    config_data = dict(gui_config)  # shallow copy
    config_data["input_dir"] = plate_job.input_dir
    if plate_job.plate_name:
        config_data["plate_name"] = plate_job.plate_name
    elif not config_data.get("plate_name"):
        config_data["plate_name"] = Path(plate_job.input_dir).name

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="aggrequant_", delete=False,
    )
    yaml.dump(config_data, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return Path(tmp.name)


def _run_single_plate(gui_config: dict, job: PlateJob):
    """Run the pipeline for one plate."""
    from aggrequant.pipeline import SegmentationPipeline
    from aggrequant.loaders.images import build_field_triplets

    job.status = "running"
    state.current_field = 0
    state.add_log("INFO", f"Starting plate: {job.plate_name or job.input_dir}")

    # Determine config path
    if job.config_path:
        config_path = Path(job.config_path)
    else:
        config_path = _build_config_from_gui(gui_config, job)

    try:
        pipeline = SegmentationPipeline(
            config_path=config_path,
            verbose=gui_config.get("verbose", True),
        )

        # Count total fields for progress
        channel_by_purpose = {}
        for ch in pipeline.config.channels:
            channel_by_purpose[ch.purpose] = ch.pattern
        triplets = build_field_triplets(
            pipeline.config.input_dir, channel_by_purpose,
        )
        state.total_fields = len(triplets)
        state.add_log("INFO", f"Found {len(triplets)} fields")

        # Monkey-patch _process_field to track progress
        original_process = pipeline._process_field

        def tracked_process(triplet):
            if state.cancel_requested:
                raise KeyboardInterrupt("Cancelled by user")
            original_process(triplet)
            state.current_field += 1
            state.add_log(
                "INFO",
                f"  Processed {triplet.well_id}/f{triplet.field_id} "
                f"({state.current_field}/{state.total_fields})",
            )

        pipeline._process_field = tracked_process
        pipeline.run()

        job.status = "done"
        state.add_log("INFO", f"Plate complete: {job.plate_name or job.input_dir}")

    except KeyboardInterrupt:
        job.status = "cancelled"
        state.add_log("WARN", f"Plate cancelled: {job.plate_name or job.input_dir}")
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        state.add_log("ERROR", f"Plate failed: {e}")
        state.add_log("ERROR", traceback.format_exc())


def start_batch(gui_config: dict, jobs: List[PlateJob]):
    """Start processing a batch of plates in a background thread."""
    if state.is_running:
        return

    state.reset()
    state.jobs = jobs
    state.total_jobs = len(jobs)
    state.is_running = True
    state.current_status = "running"

    def worker():
        try:
            for i, job in enumerate(state.jobs):
                if state.cancel_requested:
                    for remaining in state.jobs[i:]:
                        remaining.status = "cancelled"
                    break
                state.current_job_index = i
                _run_single_plate(gui_config, job)
        finally:
            state.is_running = False
            if state.cancel_requested:
                state.current_status = "cancelled"
            elif all(j.status == "done" for j in state.jobs):
                state.current_status = "done"
                state.add_log("INFO", "All plates complete!")
            else:
                state.current_status = "done"
                n_failed = sum(1 for j in state.jobs if j.status == "failed")
                if n_failed:
                    state.add_log("WARN", f"{n_failed} plate(s) failed")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def cancel_batch():
    """Request cancellation of the running batch."""
    state.cancel_requested = True
    state.add_log("WARN", "Cancellation requested...")
