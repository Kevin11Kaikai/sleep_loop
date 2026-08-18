from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata

import matplotlib
import neurolib
import numpy as np
import pandas as pd
import scipy
from neurolib.models.wc import WCModel


def distribution_version(distribution_name: str) -> str:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    model = WCModel()
    model.params.duration = 20.0
    model.run()
    output = np.asarray(model.output)
    payload = {
        "environment": {
            "sys_executable": sys.executable,
            "python_version": sys.version,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "packages": {
            "neurolib": distribution_version("neurolib"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "brainmass": distribution_version("brainmass"),
            "h5py": distribution_version("h5py"),
            "jax": distribution_version("jax"),
            "joblib": distribution_version("joblib"),
            "mne": distribution_version("mne"),
            "numba": distribution_version("numba"),
            "psutil": distribution_version("psutil"),
            "scikit_learn": distribution_version("scikit-learn"),
            "statsmodels": distribution_version("statsmodels"),
            "yaml": distribution_version("PyYAML"),
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "JAX_PLATFORM_NAME",
            )
        },
        "smoke_test": {
            "model": "neurolib.models.wc.WCModel",
            "duration_ms": 20.0,
            "output_shape": list(output.shape),
            "finite": bool(np.isfinite(output).all()),
            "status": "PASS" if np.isfinite(output).all() else "FAIL",
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
