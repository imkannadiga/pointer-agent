"""Reuses the real pointerbench-text eval.py scorer directly rather than
reimplementing its point-in-bbox / coverage-precision rules.
"""
import importlib.util
from pathlib import Path


def load_real_evaluate():
    """Downloads (if needed) and imports the real eval.py, returning its evaluate()."""
    from eval.fetch_pointerbench import fetch_eval_script

    eval_py_path = fetch_eval_script()
    spec = importlib.util.spec_from_file_location("pointerbench_real_eval", eval_py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def evaluate(gt: list[dict], preds: dict[str, dict]) -> dict:
    real_evaluate = load_real_evaluate()
    return real_evaluate(gt, preds)
