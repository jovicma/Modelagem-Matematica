"""
Exercise 8 (C2) — SIR Model: Exploring Epidemic Dynamics

Implement and explore the scaled SIR model:
    dS/dt = -R0 * S * I
    dI/dt =  R0 * S * I - I
    dR/dt =  I

and its SIRV extension with vaccination and waning immunity.

Environment:
    conda activate modmat  (numpy, scipy, matplotlib)

Usage:
    python scripts/exercises/part2/c2_sir_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# ---------------------------------------------------------------------------
# Default parameters (DO NOT MODIFY)
# ---------------------------------------------------------------------------
R0_DEFAULT = 2.0      # basic reproduction number
ALPHA_DEFAULT = 0.02  # initial infected fraction  I(0) = alpha
T_END = 20.0          # final dimensionless time
GAMMA_DEFAULT = 0.05  # waning immunity rate (SIRV)
DELTA_VAC = 0.5       # vaccination rate during campaign (SIRV)
T_VAC_START = 7.0     # vaccination start time
T_VAC_END = 15.0      # vaccination end time

# Longer horizon for the SIRV runs (waning immunity produces slow recurring
# waves that only develop well beyond T_END).
T_END_SIRV = 60.0

# Tight tolerances so the conservation invariant is limited by the model, not
# the integrator (same diagnostic used for H in the mechanics classes).
RTOL = 1e-9
ATOL = 1e-12

PESC_BLUE = "#00345a"
PESC_ORANGE = "#d68749"
PESC_GRAY = "#727176"
POSITIVE_GREEN = "#15803d"
NEGATIVE_RED = "#b91c1c"

try:
    FIG_DIR = Path(__file__).resolve().parents[3] / "Figures" / "exercises" / "part2"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
except (IndexError, OSError):
    FIG_DIR = Path(__file__).resolve().parent / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Part (a): SIR right-hand side and conservation check
# ---------------------------------------------------------------------------

def sir_rhs(t: float, y: np.ndarray, R0: float) -> list[float]:
    """Right-hand side of the scaled SIR system.

    Parameters
    ----------
    t  : current time (unused, but required by solve_ivp).
    y  : state vector [S, I, R].
    R0 : basic reproduction number.

    Returns
    -------
    [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = y
    dS = -R0 * S * I
    dI = R0 * S * I - I
    dR = I
    return [dS, dI, dR]


def solve_sir(R0: float, alpha: float, t_end: float = T_END) -> dict:
    """Solve the scaled SIR system and return solution + conservation check.

    Returns a dict with keys: 't', 'S', 'I', 'R', 'conservation_error'.
    """
    y0 = [1.0, alpha, 0.0]
    t_eval = np.linspace(0.0, t_end, 2001)
    sol = solve_ivp(sir_rhs, [0.0, t_end], y0, args=(R0,), method="RK45",
                    t_eval=t_eval, rtol=RTOL, atol=ATOL)
    S, I, R = sol.y
    total = S + I + R
    conservation_error = float(np.max(np.abs(total - (1.0 + alpha))))
    return {"t": sol.t, "S": S, "I": I, "R": R,
            "conservation_error": conservation_error}


# ---------------------------------------------------------------------------
# Part (b): Reference plots for R0 = 2 and R0 = 5
# ---------------------------------------------------------------------------

def plot_sir_reference(alpha: float = ALPHA_DEFAULT) -> None:
    """Plot S, I, R vs time for R0 = 2 and R0 = 5."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, R0 in zip(axes, [2.0, 5.0]):
        sol = solve_sir(R0, alpha)
        ax.plot(sol["t"], sol["S"], color=PESC_BLUE, label=r"$\bar{S}$")
        ax.plot(sol["t"], sol["I"], color=PESC_ORANGE, label=r"$\bar{I}$")
        ax.plot(sol["t"], sol["R"], color=POSITIVE_GREEN, label=r"$\bar{R}$")
        print(f"R0={R0}: peak I={sol['I'].max():.4f}, "
              f"final R={sol['R'][-1]:.4f}, cons_err={sol['conservation_error']:.2e}")
        ax.set_title(f"$R_0 = {R0}$", fontweight="bold", color=PESC_BLUE)
        ax.set_xlabel(r"$\bar{t}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Population fraction")
    fig.suptitle("Scaled SIR Dynamics", fontweight="bold", color=PESC_BLUE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sir_reference.pdf", dpi=300, bbox_inches="tight", transparent=True)
    print("Saved sir_reference.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part (c): R0 sweep
# ---------------------------------------------------------------------------

def r0_sweep(alpha: float = 0.01) -> None:
    """Sweep R0 from 0.5 to 6 and plot peak I and final attack rate vs R0."""
    R0_values = np.arange(0.5, 6.1, 0.5)
    peak_I = np.zeros(len(R0_values))
    attack_rate = np.zeros(len(R0_values))

    # Long horizon so the epidemic fully resolves (I -> 0) even near threshold.
    for i, R0 in enumerate(R0_values):
        sol = solve_sir(R0, alpha, t_end=120.0)
        peak_I[i] = sol["I"].max()
        S_inf = sol["S"][-1]
        attack_rate[i] = 1.0 + alpha - S_inf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(R0_values, peak_I, "o-", color=PESC_ORANGE)
    ax1.axvline(1.0, ls="--", color=PESC_GRAY, label=r"threshold $R_0=1$")
    ax1.set_xlabel(r"$R_0$")
    ax1.set_ylabel(r"peak $\bar{I}$")
    ax1.set_title("Epidemic peak intensity", fontweight="bold", color=PESC_BLUE)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(R0_values, attack_rate, "s-", color=POSITIVE_GREEN)
    ax2.axvline(1.0, ls="--", color=PESC_GRAY, label=r"threshold $R_0=1$")
    ax2.set_xlabel(r"$R_0$")
    ax2.set_ylabel(r"final attack rate $1+\alpha-\bar{S}_\infty$")
    ax2.set_title("Final epidemic size", fontweight="bold", color=PESC_BLUE)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    print("\nR0 sweep (alpha=%.2f):" % alpha)
    for R0, pI, ar in zip(R0_values, peak_I, attack_rate):
        print(f"  R0={R0:.1f}: peak I={pI:.4f}, attack rate={ar:.4f}")

    fig.suptitle("SIR: Threshold Behaviour vs $R_0$", fontweight="bold", color=PESC_BLUE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sir_r0_sweep.pdf", dpi=300, bbox_inches="tight", transparent=True)
    print("Saved sir_r0_sweep.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part (d): SIRV extension
# ---------------------------------------------------------------------------

def sirv_rhs(t: float, y: np.ndarray, R0: float, gamma: float, delta: float) -> list[float]:
    """Right-hand side of the scaled SIRV system (Class 13, Eqs. 2.43-2.46).

    State: y = [S, I, R, V]
    - delta: vaccination rate (susceptible -> vaccinated V)
    - gamma: waning immunity rate (recovered R -> susceptible S)

    dS/dt = -R0 S I + gamma R - delta S
    dI/dt =  R0 S I - I
    dR/dt =  I - gamma R
    dV/dt =  delta S
    """
    S, I, R, V = y
    dS = -R0 * S * I + gamma * R - delta * S
    dI = R0 * S * I - I
    dR = I - gamma * R
    dV = delta * S
    return [dS, dI, dR, dV]


def solve_sirv(
    R0: float,
    alpha: float,
    gamma: float = GAMMA_DEFAULT,
    delta_fn=None,
    t_end: float = T_END_SIRV,
) -> dict:
    """Solve the SIRV system.

    Parameters
    ----------
    delta_fn : callable(t) -> float, vaccination rate as function of time.
               If None, use delta = 0 (no vaccination).

    Returns a dict with keys: 't', 'S', 'I', 'R', 'V'.
    """
    if delta_fn is None:
        delta_fn = lambda t: 0.0

    def rhs(t, y):
        return sirv_rhs(t, y, R0, gamma, delta_fn(t))

    y0 = [1.0, alpha, 0.0, 0.0]
    t_eval = np.linspace(0.0, t_end, 4001)
    # max_step resolves the on/off edges of the vaccination campaign cleanly.
    sol = solve_ivp(rhs, [0.0, t_end], y0, method="RK45", t_eval=t_eval,
                    rtol=RTOL, atol=ATOL, max_step=0.05)
    S, I, R, V = sol.y
    return {"t": sol.t, "S": S, "I": I, "R": R, "V": V}


def vaccination_campaign(t: float) -> float:
    """Return delta(t): active only between T_VAC_START and T_VAC_END."""
    return DELTA_VAC if T_VAC_START <= t <= T_VAC_END else 0.0


def cumulative_infections(sol: dict, R0: float) -> float:
    """Total infections over the run: integral of the S->I flux R0*S*I dt."""
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.0 / <2.0
    flux = R0 * sol["S"] * sol["I"]
    return float(trapz(flux, sol["t"]))


def plot_sirv_comparison(R0: float = 5.0, alpha: float = ALPHA_DEFAULT) -> None:
    """Side-by-side SIRV comparison: no vaccination vs vaccination campaign."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    labels = ["No vaccination", "Vaccination campaign"]
    delta_fns = [None, vaccination_campaign]
    results = {}

    for ax, label, delta_fn in zip(axes, labels, delta_fns):
        sol = solve_sirv(R0, alpha, delta_fn=delta_fn)
        results[label] = cumulative_infections(sol, R0)
        ax.plot(sol["t"], sol["S"], color=PESC_BLUE, label=r"$\bar{S}$")
        ax.plot(sol["t"], sol["I"], color=PESC_ORANGE, label=r"$\bar{I}$")
        ax.plot(sol["t"], sol["R"], color=POSITIVE_GREEN, label=r"$\bar{R}$")
        ax.plot(sol["t"], sol["V"], color=NEGATIVE_RED, label=r"$\bar{V}$")
        if delta_fn is not None:
            ax.axvspan(T_VAC_START, T_VAC_END, color=PESC_GRAY, alpha=0.15)
        ax.set_title(label, fontweight="bold", color=PESC_BLUE)
        ax.set_xlabel(r"$\bar{t}$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Population fraction")
    fig.suptitle(f"SIRV Dynamics ($R_0 = {R0}$)", fontweight="bold", color=PESC_BLUE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sirv_comparison.pdf", dpi=300, bbox_inches="tight", transparent=True)

    c_no = results["No vaccination"]
    c_vac = results["Vaccination campaign"]
    print(f"\nSIRV cumulative infections (R0={R0}): "
          f"no vac = {c_no:.4f}, with campaign = {c_vac:.4f}, "
          f"reduction = {100*(c_no - c_vac)/c_no:.1f}%")
    print("Saved sirv_comparison.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_sir_reference()
    r0_sweep()
    plot_sirv_comparison()
