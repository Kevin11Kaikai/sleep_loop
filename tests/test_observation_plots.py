import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sleep_sbi.observation_plots import (
    PANEL_SPECS,
    SUMMARY_COLUMNS,
    plot_obs_a_recording_context,
    plot_obs_b_representative_eeg,
    plot_obs_c_psd,
    plot_obs_d_so_morphology,
    plot_obs_e_so_regularity,
    plot_obs_f_spindle_activity,
    plot_obs_g_pac,
    plot_obs_h_summary_qc,
    save_panel,
    write_observation_artifacts,
)


PLOTTERS = {
    "Obs-a": plot_obs_a_recording_context,
    "Obs-b": plot_obs_b_representative_eeg,
    "Obs-c": plot_obs_c_psd,
    "Obs-d": plot_obs_d_so_morphology,
    "Obs-e": plot_obs_e_so_regularity,
    "Obs-f": plot_obs_f_spindle_activity,
    "Obs-g": plot_obs_g_pac,
    "Obs-h": plot_obs_h_summary_qc,
}


@pytest.mark.parametrize("panel_id", PANEL_SPECS)
def test_each_observation_panel_draws_independently(
    observation_bundle, panel_id
):
    fig, panel_data = PLOTTERS[panel_id](observation_bundle)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    assert fig.axes
    assert pixels.size > 0
    assert float(np.std(pixels)) > 1.0
    assert panel_data["panel_id"] == panel_id
    assert panel_data["valid_epoch_count"] >= 0
    json.dumps(panel_data, allow_nan=False)
    plt.close(fig)


def test_all_panel_artifacts_are_openable_and_publication_safe(
    observation_bundle, tmp_path
):
    records = []
    for panel_id, plotter in PLOTTERS.items():
        fig, panel_data = plotter(observation_bundle)
        records.append(
            save_panel(
                fig,
                observation_bundle,
                panel_id,
                panel_data,
                tmp_path,
                dpi=90,
            )
        )
        plt.close(fig)

    paths = write_observation_artifacts(observation_bundle, records, tmp_path)
    assert len(records) == 8
    for _, stem in PANEL_SPECS.values():
        svg_path = tmp_path / "panels" / f"{stem}.svg"
        png_path = tmp_path / "panels" / f"{stem}.png"
        assert svg_path.stat().st_size > 1000
        assert png_path.stat().st_size > 1000
        ET.parse(svg_path)
        image = plt.imread(png_path)
        assert np.isfinite(image).all()
        assert float(np.std(image)) > 0.01

    summary = pd.read_csv(paths["observation_summary"])
    assert set(SUMMARY_COLUMNS) == set(summary.columns)
    manifest = json.loads(paths["panel_manifest"].read_text(encoding="utf-8"))
    assert manifest["panel_count"] == 8
    encoded = json.dumps(manifest, allow_nan=False)
    assert "D:\\\\" not in encoded
    assert "C:\\\\" not in encoded
    assert '"segments":' not in encoded
    assert '"raw_eeg":' not in encoded
    assert '"contains_raw_samples": false' in encoded
    assert not any(Path(path).is_absolute() for path in records[0]["artifacts"])
