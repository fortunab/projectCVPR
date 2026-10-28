"""Record numeric builds without treating a version list as an install lock."""
import argparse
import contextlib
import importlib.metadata
import io
import json
import platform
import sys
from pathlib import Path

# Torch must precede Arrow/Pandas native-library loading on the observed Windows host.
import torch
import numpy as np
import scipy
import torchvision
import pandas
import pyarrow
from threadpoolctl import threadpool_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    numpy_build, scipy_build = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(numpy_build):
        np.show_config()
    with contextlib.redirect_stdout(scipy_build):
        scipy.show_config()
    distributions = sorted(
        [{"name": d.metadata["Name"], "version": d.version}
         for d in importlib.metadata.distributions() if d.metadata["Name"]],
        key=lambda r: r["name"].lower(),
    )
    value = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_runtime": torch.__version__,
        "torchvision_runtime": torchvision.__version__,
        "numpy_runtime": np.__version__,
        "scipy_runtime": scipy.__version__,
        "pandas_runtime": pandas.__version__,
        "pyarrow_runtime": pyarrow.__version__,
        "cuda_available": torch.cuda.is_available(),
        "probe_torch_threads": torch.get_num_threads(),
        "thread_note": "Probe uses four threads. Experiment extraction/head resource settings are recorded in each experiment's own provenance.",
        "torch_build": torch.__config__.show(),
        "numpy_build": numpy_build.getvalue(),
        "scipy_build": scipy_build.getvalue(),
        "loaded_numeric_libraries": [{k: v for k, v in row.items() if k != "filepath"}
                                     for row in threadpool_info()],
        "installed_distributions": distributions,
        "scope": "Observed runtime/build information; numerical reproduction must be checked separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({k: value[k] for k in ("torch_runtime", "torchvision_runtime", "numpy_runtime", "scipy_runtime", "pandas_runtime", "pyarrow_runtime", "cuda_available")}, indent=2))


if __name__ == "__main__":
    main()
