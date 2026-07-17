# Hamiltonian Form for C6 Cycle Rydberg MIS Average Total-Time Optimization

## Problem and Objective

This result targets the Rydberg maximum independent set (MIS) problem on a 6-site ring / C6 cycle graph. The comparison baseline is `baseline_20260509`, evaluated by the mean final target-ground-subspace fidelity over independent total annealing times `T = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]`.

The acceptance criterion was that the optimized protocol family should improve the average final target fidelity over these `T` values by more than 5% relative to the baseline. This result reaches `5.526853%`.

## Base Hamiltonian

The run uses the standard analog Hamiltonian for Rydberg MIS:

```text
H0(t) = (Omega(t)/2) * sum_i sigma_i^x - Delta(t) * sum_i n_i + V * sum_(i,j in E) n_i n_j
```

where:

- `E` is the edge set of the C6 cycle graph;
- `n_i = |1_i><1_i|`；
- `Omega(t)` and `Delta(t)` are the globally controlled waveforms;
- `V` is the fixed Rydberg interaction strength;
- all candidates keep the coefficient of `V * sum_(i,j in E) n_i n_j` time independent, matching the analog Rydberg hardware constraint.

## Counterdiabatic Corrections

The successful result uses a protocol family calibrated separately for each total time, rather than a single fixed protocol shared across all `T`. The candidates include two types of counterdiabatic (CD) correction:

1. A scaled analytic first-order local CD form:

```text
H_cd(t) = alpha * f_j0(t) * sum_i sigma_i^y
```

Here `f_j0(t)` is obtained from the analytic first-order ACQC/J0 formula for smooth `Omega(t), Delta(t)`, and `alpha` is an empirical scale factor.

2. A parameterized global Y-CD form:

```text
H_cd(t) = f_y(t) * sum_i sigma_i^y
```

Here `f_y(t)` is parameterized by piecewise-linear knots and selected by candidate search. This term can be interpreted as a local Y-control approximation in the sense of a global Rydberg phase / quadrature drive.

## Physical Interpretation of the Successful Protocols

- Short times `T=8,10`: retain the baseline smooth path, but amplify the analytic first-order CD term to `alpha=1.04` to suppress diabatic transitions more strongly.
- Intermediate time `T=12`: reduce the scale to `alpha=0.6` to avoid overcompensation from an overly strong CD term at this duration.
- `T=14,16,18`: use piecewise-linear `Omega(t), Delta(t)` together with weak parameterized Y-CD, modifying the timing of the detuning zero crossing, the positive-detuning dwell, and the transverse drive to increase the final probability of landing in the degenerate MIS ground-state subspace.
- `T=20`: use the previously optimized C6 endpoint protocol with analytic J0 scaling at `alpha=0.52`.
