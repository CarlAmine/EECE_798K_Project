# Best Equations Showcase

## Scope

This package is a synthesis of the best Utah FORGE equations already produced in the repo. No new model search was run for this showcase; the results below are extracted from existing outputs.

The four showcased items are intentionally labeled differently:

- Equation (1): strongest recovered equation
- Reduced Equation (2): best final usable velocity law
- Exact-form Equation (2): closest exact RSF-looking fit, but not trusted as final because of non-identifiability
- Equation (3): weak conditional consistency check on supplied theta, not an independent recovery

## Final showcased equations

### 1. Best Equation (1): strongest recovered equation
- Source: `proposal_equation_recovery_report.md`
- Exact compact law:
  - `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V`
- One-term approximation:
  - `dtau/dt ~= 8.476432e-03*(V_drive - V)`
- Status: `best final`

### 2. Best Equation (2): best final usable velocity law
- Source: `proposal_equation_recovery_report.md`
- Final reduced RSF fallback:
  - `dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV`
- Status: `best fallback`

### 3. Closest exact RSF-looking Equation (2)
- Source: `exact_rsf_multistart_summary.json`
- Best multistart exact-form fit:
  - `dtau/dt = 2.852472e-03*(V_drive - V)`
  - `dV/dt = (1/2.519965e-02) * [tau - sigmaN*(9.042632e-12 + 7.146679e-01*log(V/V0) + 1.671089e-01*log(theta*V0/1.518258e+01))]`
  - `dtheta/dt = 1 - V*theta/1.518258e+01`
- Status: `closest exact fit but non-identifiable`

### 4. Best Equation (3) result: weak conditional consistency check
- Source: `theta_equation_consistency_report.md`
- Tiny-library consistency law:
  - `dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta`
- Implied `Dc`:
  - `Dc_hat = 1.753164e+02`
- Status: `weak consistency check`

## Compact comparison table

| label | equation text | workflow source | derivative-fit metric(s) | rollout metric(s) | divergence / stability | interpretability status | final judgment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equation (1) compact tau law | `dtau/dt = -2.557014e-02 + 3.455056e-04*V + 8.812200e-03*V_drive_minus_V` | `proposal_equation_recovery` | holdout derivative MSE `6.139149e-03` | semi-observed tau rollout MSEs: step2 `0.9796`, step7 `0.4899` | robustness check kept positive dominant `(V_drive - V)` across alternate splits | strongest recovered equation; physically compact and sign-consistent | `best final` |
| Equation (2) reduced RSF fallback | `dV/dt = 7.937418e+02 - 3.952640e+01*sigmaN - 9.497244e-01*sigmaN_logV` | `proposal_equation_recovery` | holdout derivative RMSE `2.711641e+01`, MSE `7.386781e+02`, R^2 `-98.2040` | rollout MSE `4.493035e+02`; peak timing `16.556 s`; onset timing `15.378 s` | mean stable fraction `0.006` | best final usable velocity law; reduced but physically interpretable | `best fallback` |
| Equation (2) closest exact RSF-looking fit | `dV/dt = (1/2.519965e-02) * [tau - sigmaN*(9.042632e-12 + 7.146679e-01*log(V/V0) + 1.671089e-01*log(theta*V0/1.518258e+01))]` | `exact_rsf_multistart_check` | derivative metrics not separately reported for the multistart winner | mean holdout rollout error `3.499610e+02`; peak timing `7.7535 s`; onset timing `4.6215 s` | stable fraction `0.004444`; JTJ cond `8.904456e+08`; rank `12`; confounding still `True` | exact-form and theta-active, but non-identifiable and unstable across starts | `closest exact fit but non-identifiable` |
| Equation (3) theta consistency check | `dtheta/dt = 4.084008e-05 - 5.703972e-03*Vtheta` | `theta_equation_consistency` | holdout RMSE `2.795127e-02`, R^2 `3.738273e-04` | no forward theta rollout used; consistency check only | `c0` not close to `1`; implied `Dc_hat = 1.753164e+02`; consistency strength `weak` | weak conditional check on supplied theta, not an independent state recovery | `weak consistency check` |

## Supporting notes

### Robustness confirmation
- Source: `proposal_equation_robustness_check.md`
- Across alternate splits, the compact tau law kept a positive dominant `(V_drive - V)` term.
- The reduced velocity law consistently retained negative `sigmaN*log(V/V0)`.
- The exact theta term remained non-identifiable.

### Theta-valid subset counts
- Source: `proposal_equation_recovery_report.md`
- Theta-valid steps under strict event screening: `5`
- Theta-usable steps under sample masking: `5`
- Theta-usable samples under sample masking: `15000`

### Why the exact-form Equation (2) is not the final result
- Source: `exact_rsf_multistart_check.md`
- Multistart improved convergence and made the theta term numerically active.
- But parameter estimates did not stabilize across starts.
- Near-constant `sigmaN` and parameter confounding remained.
- Final multistart conclusion: exact equation (2) still non-identifiable from current Utah FORGE data.

## Presentation-ready summary

The strongest recovered law is the compact spring-loading tau equation, which stayed stable under alternate train/holdout splits and retained the correct positive `(V_drive - V)` structure. The best final usable velocity law is not the exact RSF equation but the reduced RSF fallback, because it preserves the negative `sigmaN*log(V/V0)` structure while avoiding the non-identifiable theta term. The closest exact-form RSF fit is still useful to show because it demonstrates how far the exact latent-state recovery can be pushed, but it must be labeled as non-identifiable rather than final. The theta equation result is only a weak conditional consistency check on externally reconstructed theta, not an independent recovery of state dynamics. Taken together, the honest project claim is that Utah FORGE strongly supports Equation (1), supports a reduced RSF-style Equation (2), and does not yet support a credible exact theta-bearing recovery of Equations (2)–(3).
