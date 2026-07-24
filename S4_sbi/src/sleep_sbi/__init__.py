"""Observation-side utilities for the sleep-loop SBI workflow."""

from .observation import (
    build_observation_bundle,
    load_observation_config,
    map_raw_stage,
    metric_classification_rows,
    select_representative_epochs,
)
from .observation_plots import (
    PANEL_SPECS,
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
from .schemas import ObservationBundle, ObservationConfig, PSDResult, SummaryMetric

__all__ = [
    "ObservationBundle",
    "ObservationConfig",
    "PSDResult",
    "SummaryMetric",
    "PANEL_SPECS",
    "build_observation_bundle",
    "load_observation_config",
    "map_raw_stage",
    "metric_classification_rows",
    "plot_obs_a_recording_context",
    "plot_obs_b_representative_eeg",
    "plot_obs_c_psd",
    "plot_obs_d_so_morphology",
    "plot_obs_e_so_regularity",
    "plot_obs_f_spindle_activity",
    "plot_obs_g_pac",
    "plot_obs_h_summary_qc",
    "save_panel",
    "select_representative_epochs",
    "write_observation_artifacts",
]
