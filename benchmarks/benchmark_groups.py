"""Benchmark: GroupPrior throughput (groups per second) per mechanism.

Measures ``GroupPrior.sample`` for every mechanism at several lengths, with
all native worker threads and single-threaded (the setting for one data-loader
worker process), with and without full recipes. No model is trained.

Usage:
    uv run python benchmarks/benchmark_groups.py            # full grid
    uv run python benchmarks/benchmark_groups.py --quick    # small smoke run
"""

import argparse
import time

from synforecast import CoupledSpec, GroupPrior, RegressionSpec, VARXSpec

MECHANISMS = {
    "regression": RegressionSpec,
    "coupled": CoupledSpec,
    "varx": lambda: VARXSpec(n_shared=1, n_past=1),
}


def throughput(prior: GroupPrior, n: int, **kwargs) -> float:
    """Groups per second of one ``sample`` call (after a warm-up call)."""
    prior.sample(4, seed=99, **kwargs)
    start = time.perf_counter()
    prior.sample(n, seed=0, **kwargs)
    return n / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    args = parser.parse_args()
    lengths = [(256, 256)] if args.quick else [(512, 512), (2048, 2048), (512, 4096)]
    n = 50 if args.quick else 2000

    print(
        f"{'mechanism':12s} {'length':>12s} {'mode':>22s} {'groups/s':>10s} {'ms/group':>9s}"
    )
    for name, make in MECHANISMS.items():
        for length in lengths:
            prior = GroupPrior(mechanism=make(), length=length, freq="h")
            for mode, kwargs in [
                ("all cores", {}),
                ("1 thread", {"n_jobs": 1}),
                ("1 thread, no recipes", {"n_jobs": 1, "recipes": False}),
            ]:
                rate = throughput(prior, n, **kwargs)
                print(
                    f"{name:12s} {str(length):>12s} {mode:>22s} "
                    f"{rate:10.0f} {1e3 / rate:9.3f}"
                )


if __name__ == "__main__":
    main()
