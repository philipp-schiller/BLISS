# ── serialization ────────────────────────────────────────────────────────────
from scripts.Benchmarking.axes import NOISE_FUNCTION_CASES, SOURCE_DENSITY_CASES
from scripts.Benchmarking.ground_truth_config import GroundTruthConfig
from scripts.Benchmarking.results import SweepResult
import numpy as np
import argparse
import json
import os


from src.bayes_shape_fitting.models.shapes.ellipse import Ellipse

def save_result(result: SweepResult, save_dir: str = "results_cache") -> str:
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, result.case_name)

    np.savez_compressed(
        stem + ".npz",
        fitting_params=result.fitting_params,
        fitting_times=result.fitting_times,
        shape_gt=result.shape_gt,
        source_gt=result.source_gt,
        noise_gt=result.noise_gt,
    )

    meta = {
        "case_name": result.case_name,
        "noise_name": result.noise_name,
        "source_name": result.source_name,
        "fitter_labels": result.fitter_labels,
        "shape": result.ground_truth_config.name,
    }
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_result(stem: str) -> SweepResult:
    """Load a SweepResult from <stem>.npz + <stem>.json (no ground_truth_config)."""
    arrays = np.load(stem + ".npz")
    with open(stem + ".json") as f:
        meta = json.load(f)

    # Reconstruct a minimal stand-in for ground_truth_config (plotting only needs .shape)
    class _MinimalConfig:
        def __init__(self, shape):
            self.shape = shape

    # reconstruct noise and source from name:

    noise_function = next((item for item in NOISE_FUNCTION_CASES if item["name"] == meta["noise_name"]), None)["noise_function"]
    source_density = next((item for item in SOURCE_DENSITY_CASES if item["name"] == meta["source_name"]), None)["source_density"]

    return SweepResult(
        case_name=meta["case_name"],
        source_name=meta["source_name"],
        noise_name=meta["noise_name"],
        #ground_truth_config=_MinimalConfig(Ellipse()),
        ground_truth_config=GroundTruthConfig(shape=Ellipse(), source_density=source_density, noise_function=noise_function, shape_params=arrays["shape_gt"][0]),
        fitter_labels=meta["fitter_labels"],
        fitting_params=arrays["fitting_params"],
        fitting_times=arrays["fitting_times"],
        shape_gt=arrays["shape_gt"],
        source_gt=arrays["source_gt"],
        noise_gt=arrays["noise_gt"],
    )