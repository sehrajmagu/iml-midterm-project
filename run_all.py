"""Regenerate every result from scratch: all runs, the prediction file, and the error analysis.

Usage: python3 run_all.py [--seed 0]

Writes experimental_results.csv, predictions/submission.csv, figures/ and results/error_analysis/.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# NumPy's matrix multiply raises these on this machine during scikit-learn fits. Checks against
# an independent solver gave the same results, so only these three messages are hidden; every
# other warning still shows.
MATMUL_WARNING = r"(divide by zero|overflow|invalid value) encountered in matmul"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="seed for the inner splits, the random KFold and the bootstrap")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message=MATMUL_WARNING, category=RuntimeWarning)
    # The seed must be set before the project modules are imported, because they read it on import.
    os.environ["IML_SEED"] = str(args.seed)
    sys.path.insert(0, str(ROOT / "src"))
    import baselines
    import error_analysis
    import experiments
    import make_submission

    train = baselines.load_train()

    def log(rows):
        rows = rows if isinstance(rows, list) else [rows]
        for row in rows:
            baselines.log_run(row)
        baselines.print_rows(rows)

    steps = [(f"baseline: {name}", lambda f=f: log(f(train))) for name, f in baselines.RUNNERS.items()]
    steps += [(f"experiment: {name}", lambda f=f: log(f(train))) for name, f in experiments.RUNNERS.items()]
    steps += [("final model and predictions/submission.csv", make_submission.main),
              ("error analysis tables and figures", error_analysis.main)]

    total_start = time.perf_counter()
    for i, (name, step) in enumerate(steps, start=1):
        print(f"\n=== [{i}/{len(steps)}] {name} ===", flush=True)
        start = time.perf_counter()
        step()
        print(f"=== [{i}/{len(steps)}] {name} done in {time.perf_counter() - start:.1f}s ===", flush=True)
    print(f"\nAll {len(steps)} steps done with seed {args.seed} in {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
