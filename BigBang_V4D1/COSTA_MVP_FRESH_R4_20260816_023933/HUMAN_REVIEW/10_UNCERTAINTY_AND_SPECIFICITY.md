# Uncertainty and specificity

The empirical paired-epoch estimand was C000 MSE minus C006 MSE. Across 110 HELDOUT epochs its mean was 0.09034706523072362. The frozen 1,000-draw bootstrap (seed 26081604) produced a 95% interval of [0.07980018864484562, 0.10146456173015034], whose lower bound is strictly positive.

The relative HELDOUT NRMSE improvement was 0.46559200627853425, exceeding the frozen 0.05 margin.

Specificity remains bounded: empirical wrong-subject specificity is not identifiable from one subject; C000 is fixed rather than population-estimated; and the frozen bootstrap treats epochs as exchangeable without modeling within-night serial dependence.
