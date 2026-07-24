from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SBI_SRC = ROOT / "S4_sbi" / "src"
if str(SBI_SRC) not in sys.path:
    sys.path.insert(0, str(SBI_SRC))


@pytest.fixture(scope="session")
def observation_bundle():
    if not (ROOT / "data" / "manifest.csv").is_file():
        pytest.skip("ignored local Sleep-EDF manifest is unavailable")
    from sleep_sbi import build_observation_bundle

    return build_observation_bundle()
