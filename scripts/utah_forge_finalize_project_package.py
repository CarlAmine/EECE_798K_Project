from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import utah_forge_proposal_equation_recovery as recovery


RESULTS_DIR = recovery.RESULTS_DIR


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_text_blocks(recovery_payload: dict, robustness_payload: dict) -> dict[str, str]:
    tau_equation = recovery_payload["tau_model"]["exact_equation"]
    velocity_equation = recovery_payload["final_velocity_model"]["equation"]
    exact_model = recovery_payload["velocity_models"]["A_exact_rsf"]["best"]
    reduced_model = recovery_payload["velocity_models"]["B_reduced_rsf"]["best"]
    acoustic_model = recovery_payload["velocity_models"]["D_acoustic_augmented"]["best"]
    ident = recovery_payload["identifiability"]

    results = (
        f"The proposal-equation recovery workflow isolated compact physical laws directly in Utah FORGE units. "
        f"For the shear-stress evolution equation, the recovered form was `{tau_equation}`, with the closest one-term reduction "
        f"`{recovery_payload['tau_model']['one_term_equation']}`. This is a strong confirmation that the observed `dtau/dt` dynamics are dominated by the "
        f"loading mismatch term `(V_drive - V)`. For the velocity equation, the exact proposal form was tested explicitly, but the fitted theta coefficient "
        f"collapsed to `{exact_model['coefficients_physical'].get('sigmaN_logTheta', 0.0):.3e}` and did not remain active. The final selected model was the reduced RSF fallback "
        f"`{velocity_equation}`, which retained the negative `sigmaN*log(V/V0)` structure and therefore preserved the strongest data-supported RSF signature."
    )

    discussion = (
        f"The main scientific outcome is therefore mixed but positive. We recovered a compact governing equation for `tau` and a reduced physics-informed fallback for `V`, "
        f"while showing that exact proposal equation (2) is not credibly identifiable on the current Utah FORGE subset. The identifiability analysis points away from a coding or alignment failure "
        f"and toward structural confounding: `sigmaN` varies only weakly on the exact-train subset (`sigma_cv = {ident.get('sigma_cv', float('nan')):.3e}`), so it behaves almost like an intercept and reduces the "
        f"independent leverage needed to estimate the theta contribution. The bounded robustness check did not overturn that conclusion; it reproduced `(V_drive - V)` dominance in the tau law, preserved the reduced `log(V)` structure, "
        f"and left the theta term inactive across alternate holdout choices."
    )

    limitations = (
        f"This package should not be interpreted as an exact recovery of the full proposal velocity law. The fitted exact-RSF model did not support a stable theta term, and its rollouts were not strong enough to justify a stronger claim. "
        f"The present Utah FORGE subset also has near-constant normal stress over the theta-usable windows, which limits separation between intercept-like effects, stress dependence, and state evolution. "
        f"The acoustic feature `avg_timeshift` improved peak timing behavior in one comparison but did not add enough independent information beyond the mechanical state to replace the reduced RSF fallback as the final reported model."
    )

    conclusion = (
        f"The final project claim is that Utah FORGE data support compact, physically interpretable recovery of the shear-stress equation and a reduced RSF-consistent velocity law, but not a credible identification of the full theta-bearing proposal equation (2) on the current dataset. "
        f"That negative result is scientifically useful because it is explained by structural identifiability limits rather than an implementation bug, and because the repeated emergence of `(V_drive - V)` and `log(V)` still provides a meaningful data-driven confirmation of RSF structure."
    )
    return {"Results": results, "Discussion": discussion, "Limitations": limitations, "Conclusion": conclusion}


def build_final_summary(recovery_payload: dict, robustness_payload: dict) -> str:
    tau_equation = recovery_payload["tau_model"]["exact_equation"]
    velocity_equation = recovery_payload["final_velocity_model"]["equation"]
    ident = recovery_payload["identifiability"]
    acoustic = recovery_payload["velocity_models"]["D_acoustic_augmented"]["best"]
    blocks = final_text_blocks(recovery_payload, robustness_payload)

    theta_paragraph = (
        f"The theta term could not be identified credibly because the exact-RSF training subsets retained near-constant normal stress (`sigma_cv = {ident.get('sigma_cv', float('nan')):.3e}`), which made `sigmaN` behave almost like an intercept and structurally weakened separation of the `sigmaN*log(theta*V0/Dc)` contribution. "
        f"The hard diagnosis remained `multicollinearity_or_structural_non_identifiability`, while the bounded robustness check preserved the same conclusion across alternate holdout splits. In other words, the failure was not driven primarily by a coding bug, a gross alignment error, or over-filtering; it was driven by insufficient independent leverage in the current data geometry."
    )

    acoustic_paragraph = (
        f"The acoustic feature `avg_timeshift` was informative for timing but not for final model selection. In the main comparison it reduced peak-timing error from `{recovery_payload['velocity_models']['B_reduced_rsf']['best']['mean_peak_timing_error_s']:.3f}` s to `{acoustic['mean_peak_timing_error_s']:.3f}` s, but it worsened derivative fit and had only weak residual partial correlation with `dV/dt` beyond the mechanical baseline. "
        f"That makes it scientifically interesting as an auxiliary timing signal, but not a justified replacement for the reduced RSF fallback in the final reported governing equation."
    )

    main_claim = (
        "The main claim is that Utah FORGE supports recovery of a compact physical loading law for shear stress and a reduced RSF-consistent velocity law with explicit `log(V)` structure, while also showing that the full theta-bearing proposal equation is not identifiable on the present dataset because near-constant normal stress induces structural confounding."
    )

    lines = [
        "# Final Project Summary",
        "",
        "## Recommended equations",
        f"- Final recommended tau equation: `{tau_equation}`",
        f"- Final recommended velocity equation: `{velocity_equation}`",
        "",
        "## Why theta could not be identified",
        theta_paragraph,
        "",
        "## What the acoustic feature taught us",
        acoustic_paragraph,
        "",
        "## Main claim",
        main_claim,
        "",
        "## Report-ready text",
    ]
    for heading, text in blocks.items():
        lines.extend([f"### {heading}", text, ""])
    return "\n".join(lines).rstrip() + "\n"


def build_final_equations_txt(recovery_payload: dict) -> str:
    return "\n".join(
        [
            "Target proposal equation (1): dtau/dt = k*(V_drive - V)  [compact physical reference]",
            "Target proposal equation (2): dV/dt = beta0 + beta_tau*tau + beta_v*(sigmaN*log(V/V0)) + beta_theta*(sigmaN*log(theta*V0/Dc))",
            f"Final recommended tau equation: {recovery_payload['tau_model']['exact_equation']}",
            f"Closest one-term tau equation: {recovery_payload['tau_model']['one_term_equation']}",
            f"Final recommended velocity equation: {recovery_payload['final_velocity_model']['equation']}",
            f"Exact proposal-equation (2) attempt: {recovery_payload['velocity_models']['A_exact_rsf']['best']['equation']}",
        ]
    ) + "\n"


def build_final_model_table(recovery_payload: dict) -> pd.DataFrame:
    comparison = pd.read_csv(RESULTS_DIR / "proposal_equation_model_comparison.csv")
    role_map = {
        "A_exact_rsf": "Exact proposal equation (2) test",
        "B_reduced_rsf": "Final selected reduced RSF fallback",
        "C_local_memory": "Minimal local-memory fallback check",
        "D_acoustic_augmented": "Single-feature acoustic augmentation check",
    }
    decision_map = {
        "A_exact_rsf": "Rejected: theta term not identifiable",
        "B_reduced_rsf": "Selected final model",
        "C_local_memory": "Rejected: worse fit/stability tradeoff",
        "D_acoustic_augmented": "Rejected as final model: timing-interest only",
    }
    comparison["role"] = comparison["model"].map(role_map)
    comparison["decision"] = comparison["model"].map(decision_map)
    comparison.to_csv(RESULTS_DIR / "final_model_table.csv", index=False)
    return comparison


def build_figures_manifest() -> str:
    lines = [
        "# Final Figures Manifest",
        "",
        "- `proposal_equation_recovery_diagnostics.png`: Main four-panel figure summarizing tau fit, selected velocity fit, exact-RSF correlation structure, and model-comparison annotations.",
        "- `proposal_equation_model_comparison.csv`: Numeric table supporting the model-ladder comparison in the report.",
        "- `proposal_equation_identifiability_report.md`: Detailed exact-RSF identifiability diagnosis for equation (2).",
        "- `proposal_equation_robustness_check.md`: Bounded robustness confirmation showing that the final conclusion is stable across alternate holdout splits.",
        "",
        "Recommended figure set for the report:",
        "1. `proposal_equation_recovery_diagnostics.png` as the main summary figure.",
        "2. `proposal_equation_model_comparison.csv` converted to a report table.",
        "3. `proposal_equation_identifiability_report.md` excerpted for the exact-equation (2) failure explanation.",
    ]
    return "\n".join(lines) + "\n"


def update_recovery_report(recovery_payload: dict, robustness_payload: dict) -> None:
    blocks = final_text_blocks(recovery_payload, robustness_payload)
    tau_equation = recovery_payload["tau_model"]["exact_equation"]
    velocity_equation = recovery_payload["final_velocity_model"]["equation"]
    exact_equation = recovery_payload["velocity_models"]["A_exact_rsf"]["best"]["equation"]
    ident = recovery_payload["identifiability"]

    lines = [
        "# Proposal Equation Recovery Report",
        "",
        "## Target proposal equations",
        "- Equation (1): a compact physical loading law for shear stress, ideally dominated by `(V_drive - V)`.",
        "- Equation (2): an RSF-style velocity law with `tau`, `sigmaN*log(V/V0)`, and `sigmaN*log(theta*V0/Dc)` contributions.",
        "",
        "## Final scientific outcome",
        f"- Final tau equation: `{tau_equation}`",
        f"- Final velocity equation: `{velocity_equation}`",
        f"- Exact equation (2) test: `{exact_equation}`",
        f"- Final conclusion: `{recovery_payload['summary']['final_conclusion']}`",
        "",
        "## Results",
        blocks["Results"],
        "",
        "## Discussion",
        blocks["Discussion"],
        "",
        "## Acoustic feature result",
        f"- Acoustic feature tested: `{recovery_payload['summary']['acoustic_feature_used']}`",
        f"- Peak timing improved in the acoustic-augmented comparison, but derivative fit worsened and the feature was not selected as the final governing term.",
        "",
        "## Limitations",
        blocks["Limitations"],
        "",
        "## Why this is still a positive result",
        "Recovering the compact tau law and the reduced RSF `log(V)` structure is still an important confirmation that the Utah FORGE data contain stable, physically interpretable RSF signatures even when the full theta-bearing equation is not identifiable.",
        "",
        "## Exact equation (2) failure mode",
        f"- Hard diagnosis: `{ident['hard_diagnosis']['summary']}`",
        f"- SigmaN coefficient of variation on exact-train subset: `{ident.get('sigma_cv', float('nan')):.6e}`",
        f"- Intercept + sigmaN redundant: `{ident.get('intercept_sigmaN_redundant')}`",
        f"- Theta variation too weak after filtering: `{ident.get('theta_variation_too_weak_after_filtering')}`",
        "",
        "## Robustness confirmation",
        f"- Tau confirmation: `{robustness_payload['aggregate']['tau_equation_confirmed']}`",
        f"- Reduced log(V) confirmation: `{robustness_payload['aggregate']['reduced_logv_confirmed']}`",
        f"- Theta non-identifiability confirmation: `{robustness_payload['aggregate']['theta_nonidentifiable_confirmed']}`",
        f"- Robustness conclusion: `{robustness_payload['aggregate']['overall_conclusion']}`",
        "",
        "## Conclusion",
        blocks["Conclusion"],
        "",
        "Exact equation (2) not identifiable from current data; best fallback model is B_reduced_rsf",
    ]
    (RESULTS_DIR / "proposal_equation_recovery_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_usage_notes() -> None:
    results_readme = RESULTS_DIR / "README.md"
    results_text = results_readme.read_text(encoding="utf-8") if results_readme.exists() else "# Utah FORGE Results\n\n"
    note = (
        "\n## Proposal-Equation Workflow\n"
        "- Main recovery: `./.venv/Scripts/python -B scripts/utah_forge_proposal_equation_recovery.py`\n"
        "- Bounded robustness check: `./.venv/Scripts/python -B scripts/utah_forge_proposal_equation_robustness.py`\n"
        "- Final package assembly: `./.venv/Scripts/python -B scripts/utah_forge_finalize_project_package.py`\n"
    )
    if "## Proposal-Equation Workflow" not in results_text:
        results_readme.write_text(results_text.rstrip() + "\n" + note, encoding="utf-8")

    root_readme = REPO_ROOT / "README.md"
    root_text = root_readme.read_text(encoding="utf-8") if root_readme.exists() else "# Project\n\n"
    root_note = (
        "\n## Utah FORGE Proposal-Equation Package\n"
        "Regenerate the key proposal-equation outputs from repo root with:\n"
        "- `./.venv/Scripts/python -B scripts/utah_forge_proposal_equation_recovery.py`\n"
        "- `./.venv/Scripts/python -B scripts/utah_forge_proposal_equation_robustness.py`\n"
        "- `./.venv/Scripts/python -B scripts/utah_forge_finalize_project_package.py`\n"
    )
    if "## Utah FORGE Proposal-Equation Package" not in root_text:
        root_readme.write_text(root_text.rstrip() + "\n" + root_note, encoding="utf-8")


def main() -> None:
    recovery_payload = load_json(RESULTS_DIR / "proposal_equation_recovery.json")
    robustness_payload = load_json(RESULTS_DIR / "proposal_equation_robustness_summary.json")

    update_recovery_report(recovery_payload, robustness_payload)
    (RESULTS_DIR / "final_project_summary.md").write_text(build_final_summary(recovery_payload, robustness_payload), encoding="utf-8")
    (RESULTS_DIR / "final_equations_for_report.txt").write_text(build_final_equations_txt(recovery_payload), encoding="utf-8")
    build_final_model_table(recovery_payload)
    (RESULTS_DIR / "final_figures_manifest.md").write_text(build_figures_manifest(), encoding="utf-8")
    update_usage_notes()
    print("Final proposal-equation package artifacts written.", flush=True)


if __name__ == "__main__":
    main()
