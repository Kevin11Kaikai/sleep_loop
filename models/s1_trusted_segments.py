"""
s1_trusted_segments.py
Exploratory script: confirm Sleep-EDF data layout, file pairing,
EEG channel names, and hypnogram annotation labels.

No PSD computation, no file saving — print-only.
"""

import os
import glob
import pandas as pd
import mne

mne.set_log_level("WARNING")

# ── Step 1: read manifest and scan EDF files ──────────────────────────────────

manifest = pd.read_csv("data/manifest.csv")
print("Manifest columns:", manifest.columns.tolist())
print(manifest.head())

# scan raw directory as well (cross-check)
edf_files  = glob.glob("data/sleep-edfx-cassette/**/*.edf", recursive=True)
edf_files += glob.glob("data/sleep-edfx-cassette/*.edf")
edf_files  = sorted(set(edf_files))
print(f"\nFound {len(edf_files)} EDF files on disk:")
for f in edf_files[:10]:
    print(f"  {f}")
if len(edf_files) > 10:
    print(f"  ... ({len(edf_files) - 10} more)")

psg_files = sorted([f for f in edf_files if "PSG"       in os.path.basename(f)])
hyp_files = sorted([f for f in edf_files if "Hypnogram" in os.path.basename(f)
                                          or "Hyp"       in os.path.basename(f)])
print(f"\nPSG files : {len(psg_files)}")
print(f"Hypnogram files: {len(hyp_files)}")
print("First 3 pairs (disk scan):")
for p, h in zip(psg_files[:3], hyp_files[:3]):
    print(f"  {os.path.basename(p)}  <->  {os.path.basename(h)}")

# ── Step 2: use manifest pairs (guaranteed correct pairing) ──────────────────

print(f"\nManifest has {len(manifest)} subject entries.")
print("First 3 manifest pairs:")
for _, row in manifest.head(3).iterrows():
    psg_ok = os.path.isfile(row["psg_path"])
    hyp_ok = os.path.isfile(row["hypnogram_path"])
    print(f"  [{row['subject_id']}]  PSG {'✓' if psg_ok else '✗'}  Hyp {'✓' if hyp_ok else '✗'}")
    print(f"    PSG : {row['psg_path']}")
    print(f"    Hyp : {row['hypnogram_path']}")

# count how many pairs are actually present on disk
n_valid = sum(
    os.path.isfile(r["psg_path"]) and os.path.isfile(r["hypnogram_path"])
    for _, r in manifest.iterrows()
)
print(f"\nValid pairs on disk: {n_valid} / {len(manifest)}")

# ── Step 3: open first valid pair with MNE ───────────────────────────────────

first_valid = next(
    (r for _, r in manifest.iterrows()
     if os.path.isfile(r["psg_path"]) and os.path.isfile(r["hypnogram_path"])),
    None,
)

if first_valid is None:
    print("\n[ERROR] No valid PSG/Hypnogram pair found on disk.")
else:
    psg_path = first_valid["psg_path"]
    hyp_path = first_valid["hypnogram_path"]
    subj     = first_valid["subject_id"]

    print(f"\n── Opening subject {subj} ──")
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

    print(f"File     : {os.path.basename(psg_path)}")
    print(f"Channels : {raw.ch_names}")
    print(f"Sfreq    : {raw.info['sfreq']} Hz")
    print(f"Duration : {raw.times[-1]:.1f} s  ({raw.times[-1]/3600:.2f} h)")

    # confirm expected EEG channels
    eeg_targets = {"EEG Fpz-Cz", "EEG Pz-Oz"}
    found_eeg   = eeg_targets & set(raw.ch_names)
    missing_eeg = eeg_targets - set(raw.ch_names)
    print(f"\nExpected EEG channels: {eeg_targets}")
    print(f"  Found  : {found_eeg if found_eeg else 'none'}")
    print(f"  Missing: {missing_eeg if missing_eeg else 'none'}")

    # hypnogram annotations
    ann = mne.read_annotations(hyp_path)
    unique_desc = sorted(set(ann.description))
    print(f"\nHypnogram: {os.path.basename(hyp_path)}")
    print(f"  Total annotations : {len(ann)}")
    print(f"  Unique descriptions: {unique_desc}")

    # check for slow-wave sleep stages
    sw_stages  = [d for d in unique_desc if "3" in d or "4" in d or "N3" in d]
    rem_stages = [d for d in unique_desc if "REM" in d or "R" == d.split()[-1]]
    print(f"  Slow-wave stages   : {sw_stages  if sw_stages  else 'not found'}")
    print(f"  REM stages         : {rem_stages if rem_stages else 'not found'}")

print("\n── Summary ──────────────────────────────────────────────────────")
print(f"Manifest entries      : {len(manifest)}")
print(f"EDF files on disk     : {len(edf_files)}")
print(f"Valid PSG/Hyp pairs   : {n_valid}")
