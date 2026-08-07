"""
Run with:
    python evaluate.py                 # list all registered versions + metrics
    python evaluate.py --activate v20260728_143000   # promote a version to active
"""
import argparse
import json

from registry import activate_version, list_versions


def print_versions():
    versions = list_versions()
    if not versions:
        print("No model versions registered yet. Run train.py first.")
        return
    for v in versions:
        metrics = json.loads(v.metrics)
        flag = " <- ACTIVE" if v.is_active else ""
        print(f"{v.version}{flag}")
        print(f"  created: {v.created_at}")
        print(f"  metrics: {metrics}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activate", type=str, help="Version string to activate")
    args = parser.parse_args()

    if args.activate:
        activate_version(args.activate)
        print(f"Activated {args.activate}")
    else:
        print_versions()