# Proposal Equation Robustness Check

This was a bounded confirmation using two alternate holdout splits over the already prepared RSFit-aligned Utah FORGE segments. No new model families were introduced.

## Aggregate conclusion
- Tau equation robustness: `True`
- Reduced velocity log(V) structure retained: `True`
- Exact theta term remained non-identifiable: `True`
- Overall conclusion: `Bounded robustness check supports the fixed final conclusion.`

## Split-by-split results
### alt_split_a
- Train steps: `p5838_step8, p5838_step9, p5838_step4, p5838_step5, p5838_step10, p5838_step7`
- Holdout steps: `p5838_step2, p5838_step3`
- Tau equation: `dtau/dt = -2.205208e-02 + 3.270174e-04*V + 8.689277e-03*V_drive_minus_V`
- Tau one-term approximation: `dtau/dt ~= 8.371454e-03*(V_drive - V)`
- `(V_drive - V)` dominant and positive: `True`
- Reduced velocity equation: `dV/dt = 4.500583e+01 - 1.152033e+00*sigmaN_logV`
- Reduced `sigmaN*log(V/V0)` retained and negative: `True`
- Exact theta term active: `False` with coefficient `-7.157096e-18`
- Structural identifiability diagnosis: `combination: implementation_bug, multicollinearity_or_structural_non_identifiability`
- SigmaN coefficient of variation on exact-train subset: `6.940788e-04`

### alt_split_b
- Train steps: `p5838_step3, p5838_step9, p5838_step4, p5838_step5, p5838_step10, p5838_step2`
- Holdout steps: `p5838_step7, p5838_step8`
- Tau equation: `dtau/dt = -1.922223e-02 + 3.134296e-04*V + 8.734409e-03*V_drive_minus_V`
- Tau one-term approximation: `dtau/dt ~= 8.429970e-03*(V_drive - V)`
- `(V_drive - V)` dominant and positive: `True`
- Reduced velocity equation: `dV/dt = 8.840661e+02 - 4.428614e+01*sigmaN - 9.361022e-01*sigmaN_logV`
- Reduced `sigmaN*log(V/V0)` retained and negative: `True`
- Exact theta term active: `False` with coefficient `-1.645302e-18`
- Structural identifiability diagnosis: `combination: implementation_bug, multicollinearity_or_structural_non_identifiability`
- SigmaN coefficient of variation on exact-train subset: `6.787726e-04`

## Interpretation
- Across both alternate splits, the compact tau law remained anchored by the positive `(V_drive - V)` term.
- The reduced velocity law consistently retained the negative `sigmaN*log(V/V0)` structure, supporting a stable reduced RSF interpretation.
- The exact theta term did not reactivate under these alternate splits, and the exact-train subsets still showed near-constant `sigmaN`, supporting the original structural-identifiability conclusion rather than a coding or alignment failure.
