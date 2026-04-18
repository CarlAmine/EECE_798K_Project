# FDEM Zenodo SINDy vs LightGBM comparison

## Dataset role
- `fdem_zenodo` is a simulated granular-fault comparison dataset rather than a physical lab dataset.
- The helper scripts use the final binary column as the pointwise friction-like prediction target; this SINDy baseline now models that same `mu` signal dynamically.

## SINDy baseline
- State variables: `mu`, `Ek`
- Cycles used: `fdem_cycle_001, fdem_cycle_004, fdem_cycle_003`
- Mean rollout divergence time: `4.333333` s
- Mean rollout RMSE: `0.746964`
- Interpretation: `proxy-state model, weak physical interpretation`

## Published LightGBM benchmark
- Huang et al. reported pointwise prediction metrics on the same FDEM target, including `R^2 = 0.94` for the final optimized/statistics model and `RMSE = 0.0045` for an optimized-feature testing case.
- Source: https://www.mdpi.com/2077-1312/12/2/246 and https://zenodo.org/records/7370626

## Comparison note
- LightGBM's reported performance is a pointwise regression metric on `mu`, while SINDy's divergence time is a dynamical rollout metric.
- The comparison is therefore more direct than the earlier proxy-state run, but it still mixes regression and rollout criteria rather than one shared score.