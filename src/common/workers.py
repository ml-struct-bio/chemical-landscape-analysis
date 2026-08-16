from __future__ import annotations

import os


def safe_n_workers(reserve: int = 2, minimum: int = 1) -> int:
    """Default worker-pool size for multiprocessing.Pool-based analyses.

    os.cpu_count()/multiprocessing.cpu_count() report every core on the
    physical node, not what a SLURM job actually got allocated. On shared
    cluster nodes that mismatch causes far more worker processes to spawn
    than the job's memory allows (each worker holds its own copy of the
    RDKit/embedding data), which is what OOM-killed the PCA experiment.
    Prefer the SLURM-reported allocation, falling back to the process's
    actual CPU affinity mask, and only then to the whole-node count.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return max(minimum, int(val) - reserve)
            except ValueError:
                pass
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or minimum
    return max(minimum, n - reserve)
