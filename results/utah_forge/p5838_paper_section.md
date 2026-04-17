## Section 1: Recovered Equations

The most balanced model in the holdout study was Model B, the memory-augmented formulation. In physical units, it yielded
$$
\frac{d\tau}{dt}=1.081\times 10^{1}+1.082\times 10^{-3}V+8.479\times 10^{-3}(V_{\mathrm{drive}}-V)+7.072\,\tau-2.706\ln V+1.941\times 10^{-1}\tau\ln V-4.686\,\tau_{\mathrm{avg}}-3.169\,\tau_{\mathrm{ema}}+6.116\times 10^{-5}S,
$$
and
$$
\frac{dV}{dt}=-1.245\times 10^{3}+5.023\times 10^{-1}V+5.504(V_{\mathrm{drive}}-V)+5.823\times 10^{2}\tau+3.072\times 10^{2}\ln V-2.281\times 10^{1}\tau\ln V-1.960\times 10^{2}\tau_{\mathrm{avg}}-2.934\times 10^{2}\tau_{\mathrm{ema}}-1.294\times 10^{-1}S.
$$
The $\tau$ equation therefore retained the RSF-consistent spring-loading term $k(V_{\mathrm{drive}}-V)$ with a positive coefficient, even though additional stress, logarithmic-rate, and memory terms were still needed. The $V$ equation contained $\tau$ and $\ln V$, which corresponded to the stress-loading and direct rate-dependence terms in the RSF proposal, while $\tau_{\mathrm{avg}}$, $\tau_{\mathrm{ema}}$, and $S$ acted as empirical memory surrogates for the unobserved state variable. What remained unrecovered was a directly validated state equation for $\theta$.

## Section 2: Ablation Study

The three-model ablation showed that memory mattered, but not in a uniform way across holdout steps.

| Model | Step 2 divergence [s] | Step 7 divergence [s] |
| --- | ---: | ---: |
| A | 20.892 | 0.431 |
| B | 3.802 | 11.816 |
| C | 4.507 | 2.610 |

Model A, which used only observed variables, performed well on `step2` but collapsed almost immediately on `step7`, so its success was not robust across regimes. Model B was the most balanced model because it retained nontrivial divergence times on both holdouts rather than excelling on one and failing on the other. Model C introduced $\ln(\theta_{\mathrm{approx}})$ and $\tau\ln(\theta_{\mathrm{approx}})$, so it did recover explicit state-dependent structure, but it did not dominate the memory-surrogate model in holdout rollout. The information criteria reflected a different aspect of performance: Model C had the lowest AIC/BIC ($228559.86/228691.43$), Model B was intermediate ($276937.17/277087.80$), and Model A was highest ($278275.33/278375.74$). This indicated that Model C fit the training data more efficiently, but its generalization was weaker. Explicit $\theta$ recovery therefore remained limited by RSFit product quality rather than by the idea of adding state dependence itself.

## Section 3: Regime Dependence

The regime analysis showed that the inconsistent holdout performance was not random but regime-dependent. The two holdout steps occupied the slowest regime in the dataset: `step2` had mean $V=7.73$ and dominant frequency $f=0.0142\,\mathrm{Hz}$, while `step7` had mean $V=8.63$ and $f=0.0170\,\mathrm{Hz}$. By contrast, the nearest training regime, represented by `step3` and `step8`, was already much faster, with mean $V=23.49$ and $23.62$ and dominant frequencies $0.0431$ and $0.0433\,\mathrm{Hz}$, respectively. Faster regimes were represented by `step4/step9` and the fastest by `step5/step10`.

Both holdouts therefore lay outside the training regime envelope in mean slip velocity and dominant stress-oscillation frequency. The nearest training steps for both holdouts were `step3` and `step8`, but those were still only the closest available regimes rather than true matches. This explained why different models performed best on different holdouts: all three models were being asked to extrapolate. The result connected naturally to RSF theory, because different effective $(a-b)$ balances were expected to produce qualitatively different frictional regimes, including more stable sliding and more strongly unstable stick-slip. The data therefore suggested that a single global model was insufficient for this dataset, and regime-aware SINDy, in which separate models were trained for separate velocity-frequency regimes, was a necessary extension rather than an optional refinement.

## Section 4: Derivative and Sparsity Analysis

The derivative comparison showed that the Savitzky-Golay method with window length $15$ and polynomial order $3$ was the most robust of the tested choices. It preserved the same qualitative equation structure as the other methods, retained $\tau$-$V$ coupling and logarithmic dependence, and produced the largest mean holdout divergence time ($8.37\,\mathrm{s}$), slightly above Savitzky-Golay with window $31$ ($8.34\,\mathrm{s}$) and the five-point finite-difference stencil ($7.94\,\mathrm{s}$). The sparsity sweep then showed that no physical sparsity window existed in the tested range from $10^{-5}$ to $10^{-1}$. Across all thresholds, the structural criteria remained satisfied, but both equations kept nine active terms. This indicated that the observed stick-slip dynamics required a minimum representational complexity that additional sparsity pressure did not remove without changing the problem itself.

## Section 5: Limitations

- $\theta$ was not directly observed; the RSFit inversion produced $\theta_{\mathrm{approx}}$ with limited accuracy and only partial correspondence to the memory features.
- The training and holdout regimes did not overlap in mean slip velocity or dominant frequency, so the reported results were locally valid and partly extrapolative.
- The memory surrogates $\tau_{\mathrm{avg}}$, $\tau_{\mathrm{ema}}$, and $S$ improved rollout stability, but their physical correspondence to $\theta$ was not confirmed.
