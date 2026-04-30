# Step Variability Diagnostics

## Variability ranking
| tau_variability_rank | step_name | tau_variability_score | tau_std | tau_total_variation_per_s | tau_avg_abs_derivative | tau_range |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | p5838_step5 | 0.4948940105017468 | 0.08636173184158197 | 0.4211069887578706 | 0.4211069887620407 | 0.2876705294960402 |
| 2 | p5838_step10 | 0.4156351915016943 | 0.04883544021770621 | 0.6344146764560451 | 0.6344146764536003 | 0.24908196135179317 |
| 3 | p5838_step3 | 0.2698070065293794 | 0.08461135689719934 | 0.07175658035195581 | 0.07198326544675933 | 0.5053229849569689 |
| 4 | p5838_step8 | 0.10303275212413376 | 0.09129173148874162 | 0.07931321525006461 | 0.07955487334869621 | 0.5904969710694967 |
| 5 | p5838_step9 | 0.06480989723387459 | 0.07407781911297252 | 0.18597512598678595 | 0.19285706694619634 | 0.4219456518493736 |
| 6 | p5838_step4 | -0.36018968038072074 | 0.05797848126103736 | 0.27300961195458545 | 0.2909998861689849 | 0.2908821308011955 |
| 7 | p5838_step2 | -0.4336994395938489 | 0.06516379202768303 | 0.03443267140810337 | 0.03444629770930539 | 0.6185860160109691 |
| 8 | p5838_step7 | -0.5542897379162589 | 0.06670320468113676 | 0.027242649573216562 | 0.027258029780623156 | 0.33588286241956844 |

## Holdout focus
| step_name | tau_variability_rank | tau_variability_score | tau_std | tau_total_variation_per_s | tau_range | tau_std_ratio_prepared_over_raw | tau_total_variation_ratio_prepared_over_raw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p5838_step2 | 7 | -0.4336994395938489 | 0.06516379202768303 | 0.03443267140810337 | 0.6185860160109691 | 0.9342298552595482 | 0.06306590221443475 |
| p5838_step7 | 8 | -0.5542897379162589 | 0.06670320468113676 | 0.027242649573216562 | 0.33588286241956844 | 0.983952506501437 | 0.035393876984497245 |

## Preprocessing audit
### p5838_step2
- Raw rows: `70403`
- Rows after downsampling before smoothing: `3000`
- Rows after preparation: `3000`
- Sample-theta-ok fraction: `1.000`
- Prepared/raw tau std ratio: `0.934`
- Prepared/raw tau total-variation ratio: `0.063`
- Prepared/raw dtau std ratio: `0.540`
### p5838_step7
- Raw rows: `58866`
- Rows after downsampling before smoothing: `3000`
- Rows after preparation: `3000`
- Sample-theta-ok fraction: `1.000`
- Prepared/raw tau std ratio: `0.984`
- Prepared/raw tau total-variation ratio: `0.035`
- Prepared/raw dtau std ratio: `0.330`

