# Project Summary: Physics-Informed Sparse Identification of Stick-Slip Friction Dynamics

## What This Project Is (and Is Not)

This project is about **recovering interpretable governing equations** from laboratory stick-slip friction data. It is **not** about predicting earthquakes.

The system studied is a laboratory rock friction experiment at Utah FORGE (experiment p5838), where a rock sample is loaded by a spring-slider apparatus until it undergoes repeated stick-slip instabilities. The goal is to identify the ordinary differential equations (ODEs) that govern the time evolution of shear stress (τ) and slip velocity (V) during these events — from data alone, using sparse regression.

This is a **governing equation recovery** problem, not a forecasting problem.

---

## The Physical System

### Utah FORGE p5838

Utah FORGE is a geothermal research site. Experiment p5838 uses a triaxial apparatus to compress a rock sample under controlled normal stress (σ_N) while applying a driving velocity (V_drive). The sample repeatedly sticks (elastic loading) and slips (dynamic weakening), producing a sawtooth-like shear stress signal.

Each stick-slip cycle consists of:
1. **Loading phase:** τ increases as the driving plate moves; V ≈ 0
2. **Nucleation phase:** τ approaches peak; V begins to accelerate
3. **Slip phase:** τ drops rapidly; V spikes
4. **Restrengthening phase:** τ recovers; V decays back to ≈ 0

---

## Notation and Variables

| Symbol | Description | Units |
|--------|-------------|-------|
| τ | Shear stress measured on the fault | MPa |
| V | Slip velocity of the fault surface | µm/s |
| V_drive | Machine driving velocity (controlled) | µm/s |
| σ_N | Normal stress (controlled, approximately constant) | MPa |
| θ | RSF state variable (contact age / memory) | s |
| S | Cumulative slip displacement | µm |
| τ_avg | Rolling mean of τ over a window (surrogate for steady-state) | MPa |
| τ_ema | Exponential moving average of τ (memory surrogate) | MPa |
| k | Machine stiffness (spring constant in spring-slider model) | MPa/(µm/s) |
| a, b | RSF direct effect and evolution effect coefficients | dimensionless |
| D_c | RSF critical slip distance | µm |
| µ₀ | Reference friction coefficient | dimensionless |

---

## Rate-and-State Friction (RSF) Equations

RSF is the standard physics model for laboratory stick-slip. It consists of:

### Stress law (spring-slider):
```
dτ/dt = k · (V_drive − V)
```
This is a mechanical equation relating stress rate to the velocity difference between the driving plate and the fault.

### Friction law (Dieterich-Ruina):
```
τ = σ_N · [µ₀ + a·ln(V/V₀) + b·ln(θ·V₀/D_c)]
```
This relates shear stress to friction, which depends on velocity and state.

### State evolution law (Dieterich aging law):
```
dθ/dt = 1 − θ·V/D_c
```
The state variable θ evolves with slip and time. At steady-state: θ_ss = D_c/V.

### Velocity law (derived from combining the above):
```
dV/dt = [dτ/dt − σ_N · b/θ · dθ/dt] / [σ_N · a/V]
```
Or equivalently, in reduced form:
```
dV/dt ≈ f(τ, V, θ)
```

---

## What We Can Observe

- τ (measured directly)
- V (measured or derived from displacement)
- V_drive (controlled, known)
- S (cumulative slip, from displacement)

What we **cannot directly observe:**
- θ (state variable — must be inferred)
- a, b, D_c (RSF parameters — must be estimated)

---

## Why This Is Hard

1. **θ is not observed.** The state variable must be reconstructed or estimated.
2. **The velocity law has logarithmic nonlinearity.** Standard polynomial SINDy libraries are a poor fit.
3. **Regime heterogeneity.** Different stick-slip cycles have different dynamic properties, making cross-cycle generalization hard.
4. **Derivative noise.** Computing dτ/dt and dV/dt from noisy measurements amplifies errors.
5. **Parameter non-identifiability.** Multiple parameter combinations can fit the data equally well.

---

## Final Honest Assessment

- **What was successfully recovered:** The stress law dτ/dt ≈ k(V_drive − V) is robustly identified across splits and model families.
- **What was partially recovered:** Logarithmic velocity structure is present in the velocity equation but coefficients are split-dependent.
- **What was not recovered:** True θ dynamics are not identifiable from observed data alone.
- **Exact RSF fitting:** Parameter estimation is unstable; multiple local optima exist.
- **Regime mismatch:** The most important explanation for why holdout performance differs from training performance.

---

## Why Utah FORGE is the Primary Dataset

- Controlled laboratory conditions → cleaner ODE recovery
- Well-instrumented → τ, V, V_drive all measured
- Clear stick-slip structure → natural segmentation into cycles
- Physical parameters are known approximately → useful for validation

See [`datasets.md`](datasets.md) for comparison with LANL and PANGAEA datasets.
