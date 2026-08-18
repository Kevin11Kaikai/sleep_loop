# Independent verification

G7 result: `PASS_INDEPENDENT_REPRODUCTION`; verifier status: `VERIFIED`; `MATERIAL_DISAGREEMENT = FALSE`.

Before seeing worker outcomes, the verifier froze a preinterpretation and independently reproduced frozen identities, the raw-derived FIT/HELDOUT payloads, partition digest and denominators, C006 and both affine calibrations, C000, synthetic and empirical metrics, thresholds, and gate decisions.

After the verifier-output freeze, exactly six permitted worker artifacts were opened and compared. All 6/6 were byte-identical, with zero mismatches: synthetic report/manifest, FIT selection freeze/manifest, and HELDOUT result/manifest. The repeated core digest was `A3CE6D2D01FDEDBDFBC4C19E4D685F4CBB5CC67DC6CBF958AD395A02EB9A97D7`.

The verifier used the same frozen code, data, and environment under blind independent execution. This is independent reproduction, not an independent implementation.
