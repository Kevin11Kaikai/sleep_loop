# Official data provenance — frozen before local raw access

Source: PhysioNet, **Sleep-EDF Database Expanded v1.0.0**, DOI `10.13026/C2X676`.

Official cassette index: `https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/`  
Official checksums: `https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt`

Frozen pair:

| Role | Official relative path | Official bytes | Official SHA-256 |
|---|---|---:|---|
| Night-1 PSG | `sleep-cassette/SC4001E0-PSG.edf` | 48,338,048 | `2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131` |
| Night-1 annotations | `sleep-cassette/SC4001EC-Hypnogram.edf` | 4,620 | `a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996` |

PhysioNet describes PSG files as whole-night recordings with Fpz-Cz and Pz-Oz EEG and the corresponding Hypnogram files as manual R&K sleep-stage annotations. The R4 mapping of legacy stages 3 and 4 to N3 is prospectively frozen in the builder specification.

No local EDF was read, listed, previewed, or hashed in producing this receipt. Local identity must later match both official byte count and SHA-256 before any MNE scientific read.
