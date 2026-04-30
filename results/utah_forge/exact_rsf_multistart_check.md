# Exact RSF Multistart Check

- Settings: `max_nfev=300`, `n_starts=4`
- Best run success: `True`
- Best run message: ``ftol` termination condition is satisfied.`
- Best run cost: `2.483117e+03`
- Best run mean holdout rollout error: `3.499610e+02`

## Requested checks
- Fit converges more cleanly: `True`
- Theta term becomes meaningfully active: `True`
- Parameter estimates stabilize across starts: `False`
- Identifiability metrics improve: `True`
- Final scientific conclusion changes: `False`

## Baseline vs best multistart
- Baseline success / nfev / cost: `False` / `80` / `4.042904e+03`
- Best success / nfev / cost: `True` / `37` / `2.483117e+03`
- Baseline JTJ condition number: `1.621294e+17`
- Best JTJ condition number: `8.904456e+08`
- Baseline JTJ rank: `10`
- Best JTJ rank: `12`
- SigmaN too constant in best run: `True`
- Parameter confounding in best run: `True`

## Final statement
- `Equation (2) remains non-identifiable even after direct latent-state inverse fitting with bounded multistart optimization.`
