# C1 Readiness Checklist

## Mechanistic and computational foundation

- [x] Frozen R4/M1 candidate C006 is identified and hash checked.
- [x] Streaming control implementation matches the frozen zero-control model within `1e-12`.
- [x] Controller simulations are numerically stable.
- [x] T1–T8 measurements and the response score are prospectively frozen for the final run.
- [x] Strict yoked phase negative control is valid.

## Exploratory C1 proxy signal

- [ ] A controller improves T2 in at least 2/3 evaluation seeds.
- [ ] Median T2 improvement is at least 10%.
- [ ] Median composite score R is at least 10%.
- [ ] T5 and T6 median effects are non-negative.
- [ ] Phase-locked adaptive control beats matched phase-shifted control by at least 5 percentage points.
- [ ] A valid R4 PID comparator exists, or the absence is explicitly accepted.

## Formal C1 evidence

- [ ] The controlled intervention has real pre/post patient data.
- [ ] Data and analysis are independent of development/tuning.
- [ ] Clinical or sleep-quality endpoints are prospectively defined.
- [ ] Formal authorization and confirmatory evidence requirements are satisfied.

## Current determination

`EXPLORATORY_C1_PROXY_MVP_MET = false`

`C1_CERTIFIED = false`
