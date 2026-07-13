# Inverted Pendulum on a Cart — Comparative Control Study

A from-scratch implementation and benchmark of four control strategies — **manually-tuned PID**, **differential-evolution-optimized PID**, **LQR**, and **constrained MPC** — on the classical nonlinear cart-pole system, built entirely in Python.

![Comparative results](plots/Comparative_Overview.png)

## Why this project

Most cart-pole demos either (a) only implement one controller, or (b) test controllers only against a linearized model. This project does neither:

- The **plant is fully nonlinear** — no small-angle approximation — integrated with an adaptive-step RK45 solver (`scipy.integrate.solve_ivp`), so every controller is tested against the true dynamics, not a simplified stand-in.
- **Four controllers spanning three control paradigms** (classical PID, black-box-optimized PID, and model-based optimal/predictive control) are implemented and benchmarked in one consistent framework, with identical initial conditions and identical performance metrics.
- One controller is tuned using **differential evolution**, a population-based black-box optimizer — used here as a first hands-on introduction to learning-based control, with the physics simulator embedded directly in the optimizer's cost function.

## System

Standard cart-pole benchmark: a cart of mass `M` on a horizontal rail, actuated by force `u`, carrying an inverted pendulum of point mass `m` and length `l`. State: `[x, ẋ, θ, θ̇]`.

Nonlinear equations of motion (derived and implemented exactly, no linearization in the plant itself):

```
ẍ = [ u + m·sin(θ)·(l·θ̇² − g·cos(θ)) ] / [ M + m·sin²(θ) ]
θ̈ = ( g·sin(θ) − ẍ·cos(θ) ) / l
```

## Controllers

| Controller | File | Approach |
|---|---|---|
| Manual PID | `PID.py` | Two coupled PD/PID loops (angle + position) with clamped anti-windup, hand-tuned gains |
| ML-Optimized PID | `PID_ml.py` | Same PID structure, gains found via `scipy.optimize.differential_evolution` against a closed-loop simulation cost function |
| LQR | `LQR.py` | Jacobian linearization about the upright equilibrium + continuous-time Algebraic Riccati Equation (`solve_continuous_are`) for optimal full-state feedback |
| MPC | `MPC.py` | Discretized linear model, finite-horizon (N=25) constrained QP solved every timestep with `cvxpy` + OSQP, explicit actuator saturation |

## Results

Benchmark: all controllers start from a `θ₀ = 0.2 rad` disturbance and are simulated for 6 s (`dt = 0.02 s`).

| Controller | Synthesis Time | Exec. Time (300 steps) | Overshoot | Settling Time | Control Effort (Σu²) |
|---|---|---|---|---|---|
| Manual PID | ~0 s | 0.049 s | 0.7 % | 0.80 s | 8.04 × 10⁵ |
| ML-Optimized PID | ~48 s (offline) | 0.046 s | 0.0 % | 0.88 s | 1.20 × 10³ |
| LQR | 0.002 s | 0.050 s | 0.0 % | 1.92 s | 2.82 × 10² |
| MPC (N=25) | 0.001 s | 23.8 s | 0.0 % | 1.88 s | 1.82 × 10² |

**Key takeaway:** the ML-optimized PID controller matches the manual PID's settling speed with **~670× less control effort**, while LQR and MPC achieve the smoothest, lowest-effort responses at the cost of a slower settling time — a direct, measured illustration of the synthesis-cost vs. runtime-cost vs. performance trade-off across classical, black-box, and optimal control approaches.

See [`Inverted_Pendulum_Control_Report.pdf`](Inverted_Pendulum_Control_Report.pdf) for the full write-up, derivations, plots, and discussion.

## Repository structure

```
cart_pendulum/
├── physics_env.py   # Nonlinear plant model, RK45 integration
├── PID.py           # PID class (anti-windup) + dual-loop CartPolePID
├── PID_ml.py         # Differential-evolution PID gain tuning
├── LQR.py            # Linearization + Riccati-based LQR
├── MPC.py            # Discretized, constrained MPC (cvxpy/OSQP)
└── main.py           # Benchmark harness: runs all controllers, computes metrics, plots results
plots/
└── *.png             # Comparative and per-controller response plots
Inverted_Pendulum_Control_Report.pdf   # Full project report
```

## Running it

```bash
pip install numpy scipy matplotlib cvxpy
cd cart_pendulum
python main.py
```

This trains the ML-PID controller (~1 minute), runs all four controllers on the disturbance-rejection benchmark, prints a metrics table, and shows comparative + per-controller plots.

## Future work

- Test MPC's constraint-handling advantage under a larger disturbance or tighter actuator limit, where LQR's linearization should break down but MPC should still recover.
- Add sensor noise and a state estimator (Kalman filter) — all four controllers currently assume perfect full-state feedback.
- Try a genuinely nonlinear MPC formulation to quantify the performance left on the table by linearizing.
- Implement gain scheduling to interpolate PID gains across operating regions, improving robustness for large-angle disturbances where fixed linear gains are known to degrade.

## Author

Poorvansh — B.Tech Aerospace Engineering, IIT Bombay
