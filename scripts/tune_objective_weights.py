#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for SAC objective weights.
Evaluates speed MSE on validation trips.
"""
import optuna
import numpy as np
from pathlib import Path
import subprocess
import yaml

# Paths

CONFIG_PATH = Path("training/config.yaml")
EVAL_SCRIPT = Path("evaluation/eval_offline.py")

def get_reference_dataset():
    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    # Try to get reference_dataset from top-level or env config
    ref = config.get("reference_dataset")
    if not ref:
        ref = config.get("env", {}).get("reference_dataset")
    if not ref:
        # fallback to synthetic default
        ref = "data/processed/synthetic/conditional_flow_large.pt"
    return ref

# Default weights and ranges
WEIGHT_RANGES = {
    "track_weight": (0.5, 2.0),
    "jerk_weight": (0.01, 0.5),
    "action_weight": (0.01, 0.5),
    "voltage_weight": (0.001, 0.1),
    "brake_weight": (0.001, 0.1),
    "smooth_action_weight": (0.01, 0.5),
}


def update_config(weights: dict):
    """Update config.yaml with new weights."""
    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f)
    for k, v in weights.items():
        if "reward" in config["env"]:
            config["env"]["reward"][k] = float(v)
        else:
            config["env"][k] = float(v)
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f)



def evaluate_policy() -> float:
    """Run evaluation script and return speed MSE (lower is better)."""
    reference_dataset = get_reference_dataset()
    result = subprocess.run([
        "python", str(EVAL_SCRIPT),
        "--reference-dataset", reference_dataset,
        "--metric", "speed_mse"
    ], capture_output=True, text=True)
    try:
        mse = float(result.stdout.strip().splitlines()[-1])
    except Exception:
        mse = np.inf
    return mse


def objective(trial):
    weights = {k: trial.suggest_float(k, *v) for k, v in WEIGHT_RANGES.items()}
    update_config(weights)
    # Optionally retrain or reload checkpoint here
    mse = evaluate_policy()
    return mse


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best weights:", study.best_params)
    print("Best speed MSE:", study.best_value)
    # Save results
    with open("tuning_results.yaml", "w") as f:
        yaml.safe_dump({"best_params": study.best_params, "best_value": study.best_value}, f)


if __name__ == "__main__":
    main()
