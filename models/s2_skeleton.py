"""
s2_skeleton.py
Load the HCP structural connectome from neurolib and save Cmat / Dmat to disk.
No simulation, no EEG — structural skeleton only.
"""

import os
import numpy as np
from neurolib.utils.loadData import Dataset

# ── 1. Load HCP connectome ──────────────────────────────────────────────────
ds = Dataset("hcp")
Cmat = ds.Cmat   # (80, 80) connection strength, max-normalised
Dmat = ds.Dmat   # (80, 80) fibre length in mm

# ── 2. Print basic statistics ───────────────────────────────────────────────
for name, mat in [("Cmat", Cmat), ("Dmat", Dmat)]:
    nonzero = np.count_nonzero(mat)
    print(f"{name}:")
    print(f"  shape      : {mat.shape}")
    print(f"  non-zero   : {nonzero}")
    print(f"  min / max  : {mat.min():.4f} / {mat.max():.4f}")
    print(f"  mean (all) : {mat.mean():.4f}")

N = Cmat.shape[0]
print(f"\nN = {N} brain regions (AAL2 atlas, HCP dataset)")

# ── 3. Save to data/ ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
np.save("data/cmat.npy", Cmat)
np.save("data/dmat.npy", Dmat)

# ── 4. Confirmation ─────────────────────────────────────────────────────────
print(f"\nStructural skeleton ready: Cmat {Cmat.shape}, Dmat {Dmat.shape}")
