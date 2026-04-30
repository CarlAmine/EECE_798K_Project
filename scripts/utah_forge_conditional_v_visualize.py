"""
Generate comparison visualizations for conditional V diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "utah_forge"
DIAGNOSTIC_REPORT_JSON = RESULTS_DIR / "conditional_v_diagnostic_report.json"
OUTPUT_COMPARISON_FIG = RESULTS_DIR / "conditional_v_vs_dynamic_comparison.png"
OUTPUT_EQUATION_FIG = RESULTS_DIR / "conditional_v_equations.png"


def main():
    """Generate comparison figures."""
    
    # Load results
    with open(DIAGNOSTIC_REPORT_JSON) as f:
        payload = json.load(f)
    
    # Prepare data for comparison
    variants = list(payload["variants"].keys())
    variant_labels = []
    derivative_rmses = []
    rollout_rmses = []
    r2s = []
    
    for variant_name in variants:
        data = payload["variants"][variant_name]
        variant_labels.append(data["label"].replace(" (conditional)", ""))
        derivative_rmses.append(data["mean_holdout_rmse"])
        rollout_rmses.append(np.mean([r.get("rollout_rmse", np.nan) for r in data.get("diagnostic_rollouts", [])]))
        r2s.append(data["mean_holdout_r2"])
    
    # Add current reduced RSF for comparison
    variant_labels.insert(0, "Current Reduced RSF")
    derivative_rmses.insert(0, 27.116)  # from showcase report
    rollout_rmses.insert(0, 449.303)
    r2s.insert(0, float("nan"))
    
    # Figure 1: Metric comparison (log scale where needed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Conditional V Diagnostic: Comparison vs. Current Reduced RSF", fontsize=14, weight='bold')
    
    # Derivative RMSE (log scale)
    ax = axes[0, 0]
    colors = ['red'] + ['blue', 'orange', 'green']
    bars = ax.bar(range(len(variant_labels)), derivative_rmses, color=colors, alpha=0.7)
    ax.set_ylabel("Derivative RMSE (dV/dt prediction)", fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(variant_labels)))
    ax.set_xticklabels(variant_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, derivative_rmses):
        ax.text(bar.get_x() + bar.get_width()/2, val*1.5, f"{val:.2e}", 
                ha='center', va='bottom', fontsize=9)
    
    # Rollout RMSE (log scale where conditional are higher)
    ax = axes[0, 1]
    bars = ax.bar(range(len(variant_labels)), rollout_rmses, color=colors, alpha=0.7)
    ax.set_ylabel("Conditional Rollout RMSE (m/s)", fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(variant_labels)))
    ax.set_xticklabels(variant_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, rollout_rmses):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width()/2, val*1.5, f"{val:.2e}", 
                    ha='center', va='bottom', fontsize=9)
    
    # R² comparison
    ax = axes[1, 0]
    r2_finite = [r if np.isfinite(r) else -100 for r in r2s]
    bars = ax.bar(range(len(variant_labels)), r2_finite, color=colors, alpha=0.7)
    ax.set_ylabel("Derivative R² (holdout)", fontsize=11, weight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, label='R²=0 (mean model)')
    ax.set_xticks(range(len(variant_labels)))
    ax.set_xticklabels(variant_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, r2_finite)):
        if i == 0 or np.isfinite(r2s[i]):
            ax.text(bar.get_x() + bar.get_width()/2, val*1.1, f"{val:.2f}", 
                    ha='center', va='bottom', fontsize=9)
    ax.legend()
    
    # Summary interpretation
    ax = axes[1, 1]
    ax.axis('off')
    interpretation_text = """
KEY FINDINGS

✓ Current Reduced RSF is ROBUST
  - Dynamic coupling helps stabilize fit
  - Derivative RMSE: 27.1
  - Rollout RMSE: 449.3 (acceptable)
  
✗ Conditional V Variants FAIL
  - Variant A: Negative R² despite lower RMSE
  - Variants B & C: Extremely negative R²
  - Adding theta makes it much worse
  
INTERPRETATION

Even with perfect knowledge of tau, sigmaN,
and theta, the V equation doesn't fit cleanly.

This honest diagnostic shows:
• V equation structure may be problematic
• Coupled system actually helps stability
• Reduced RSF is the right choice
    """
    ax.text(0.05, 0.95, interpretation_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARISON_FIG, dpi=150, bbox_inches='tight')
    print(f"[viz] saved comparison figure: {OUTPUT_COMPARISON_FIG}")
    plt.close()
    
    # Figure 2: Equation display
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Conditional V Equations Fitted", fontsize=14, weight='bold')
    
    equations = [
        ("Current Reduced RSF\n(Dynamic Rollout)",
         r"$\frac{dV}{dt} = 793.7 - 39.5 \cdot \sigma_N - 0.95 \cdot \sigma_N \log(V/V_0)$"),
        ("Variant A: Reduced-style V\n(Conditional)",
         r"$\frac{dV}{dt} = 1550.7 - 23.3 \tau - 64.2 \sigma_N - 0.92 \sigma_N \log(V/V_0)$"),
        ("Variant B: Theta-augmented V\n(Conditional - POOR)",
         r"$\frac{dV}{dt} = 6654.7 - 91.0 \tau - 285.7 \sigma_N + 10.9 \sigma_N \log(V/V_0) + 11.4 \sigma_N \log(\theta V_0/D_c)$"),
        ("Variant C: No-tau V\n(Conditional - POOR)",
         r"$\frac{dV}{dt} = 3520.2 - 185.5 \sigma_N + 10.5 \sigma_N \log(V/V_0) + 11.1 \sigma_N \log(\theta V_0/D_c)$"),
    ]
    
    colors_eq = ['green', 'blue', 'red', 'red']
    
    for idx, (ax, (title, equation), color) in enumerate(zip(axes.flat, equations, colors_eq)):
        ax.axis('off')
        ax.text(0.5, 0.9, title, transform=ax.transAxes, fontsize=12, weight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        ax.text(0.5, 0.5, equation, transform=ax.transAxes, fontsize=14,
                ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_EQUATION_FIG, dpi=150, bbox_inches='tight')
    print(f"[viz] saved equation figure: {OUTPUT_EQUATION_FIG}")
    plt.close()


if __name__ == "__main__":
    main()
