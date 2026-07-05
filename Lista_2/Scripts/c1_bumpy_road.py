"""
Exercise 7 (C1) — The Bumpy Road: Full Pipeline

Implement a centered finite difference solver for the vehicle-suspension ODE
    m u'' + b u' + k u = F(t)
and apply it to simulate the response to a road profile.

Environment:
    conda activate modmat  (numpy, scipy, matplotlib)

Usage:
    python scripts/exercises/part2/c1_bumpy_road.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Road data
# ---------------------------------------------------------------------------
# NOTE: the .npy referenced below is no longer reachable (the file does not
# exist in the hplgit/bumpy repository, and Langtangen's original data URL at
# hplbit.bitbucket.org is offline). We therefore SYNTHESIZE a clean road
# profile analytically (see synthetic_road). load_road_data still attempts the
# URL first, so dropping a real data file in place needs no other change.
ROAD_DATA_URL = (
    "https://github.com/hplgit/bumpy/raw/master/doc/src/fig-bumpy/bumpy_road.npy"
)

# ---------------------------------------------------------------------------
# Default physical parameters (DO NOT MODIFY)
# ---------------------------------------------------------------------------
M_DEFAULT = 60.0    # vehicle mass [kg]
K_DEFAULT = 60.0    # spring constant [N/m]
B_DEFAULT = 80.0    # damping coefficient [Ns/m]
V_DEFAULT = 5.0     # vehicle speed [m/s]
U0_DEFAULT = 0.0    # initial displacement [m]
V0_DEFAULT = 0.0    # initial velocity [m/s]

# Road geometry (spatial mesh for the synthetic profile)
X_MAX_DEFAULT = 40.0   # road length [m]
DX_DEFAULT = 0.02      # spatial resolution [m]

PESC_BLUE = "#00345a"
PESC_ORANGE = "#d68749"
PESC_GRAY = "#727176"

try:
    FIG_DIR = Path(__file__).resolve().parents[3] / "Figures" / "exercises" / "part2"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
except (IndexError, OSError):
    FIG_DIR = Path(__file__).resolve().parent / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Part (a): Centered finite difference solver
# ---------------------------------------------------------------------------

def solve_oscillator(
    t: np.ndarray,
    F: np.ndarray,
    m: float,
    b: float,
    k: float,
    u0: float,
    v0: float,
) -> np.ndarray:
    """Solve m u'' + b u' + k u = F(t) by centered finite differences.

    Discretising u'' by [D_t D_t u]^n and u' by the centered first difference
    (u^{n+1} - u^{n-1})/(2 dt) and isolating u^{n+1} gives:

        u^{n+1} = ( 2 m u^n + (b dt/2 - m) u^{n-1}
                    + dt^2 (F^n - k u^n) ) / (m + b dt/2).

    The three-level recurrence needs u^0 and a first value u^1. From a Taylor
    expansion of u^1 combined with the ODE evaluated at t=0
    (u''(0) = (F^0 - b v0 - k u0)/m):

        u^1 = u0 + dt v0 + dt^2/(2 m) (-b v0 - k u0 + F^0).

    Parameters
    ----------
    t : 1-D array of shape (N,), uniformly spaced time points.
    F : 1-D array of shape (N,), forcing evaluated at each t[n].
    m, b, k : physical parameters.
    u0, v0 : initial displacement and velocity.

    Returns
    -------
    u : 1-D array of shape (N,), displacement at each t[n].
    """
    m = float(m)
    b = float(b)
    N = t.size - 1                 # number of time intervals
    dt = t[1] - t[0]
    u = np.zeros(N + 1)

    denom = m + 0.5 * b * dt
    u[0] = u0
    u[1] = u[0] + dt * v0 + (dt ** 2 / (2.0 * m)) * (-b * v0 - k * u[0] + F[0])
    for n in range(1, N):
        u[n + 1] = (
            2.0 * m * u[n]
            + (0.5 * b * dt - m) * u[n - 1]
            + dt ** 2 * (F[n] - k * u[n])
        ) / denom
    return u


def verify_solver() -> None:
    """Check solve_oscillator against u(t) = cos(omega * t) for the undamped case."""
    omega = 2.0
    m, b, k = 1.0, 0.0, omega**2
    t = np.linspace(0, 5, 2001)
    F = np.zeros_like(t)
    u = solve_oscillator(t, F, m, b, k, u0=1.0, v0=0.0)
    u_exact = np.cos(omega * t)
    dt = t[1] - t[0]
    max_err = np.max(np.abs(u - u_exact))
    print(f"Verification: dt = {dt:.4f}, max error = {max_err:.2e}")
    # Second-order convergence: error ~ dt^2; expected ~1e-5 for dt=5e-3
    assert max_err < 1e-3, f"Solver error too large: {max_err:.2e}"
    print("Verification PASSED.")


# ---------------------------------------------------------------------------
# Part (b): Load road profile and assemble forcing
# ---------------------------------------------------------------------------

def synthetic_road(
    x_max: float = X_MAX_DEFAULT, dx: float = DX_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic clean road profile: two smooth bumps and one dip.

    A superposition of Gaussians reproduces the reference profile used in the
    course slides: an up-bump near x=8 m, a shallow dip near x=19 m, and a
    second up-bump near x=30 m. Heights are of order a few centimetres.
    """
    x = np.arange(0.0, x_max + dx, dx)
    h = (
        0.050 * np.exp(-((x - 8.0) / 2.0) ** 2)     # first bump, ~5 cm
        - 0.030 * np.exp(-((x - 19.0) / 2.5) ** 2)  # dip, ~3 cm
        + 0.055 * np.exp(-((x - 30.0) / 2.2) ** 2)  # second bump, ~5.5 cm
    )
    return x, h


def load_road_data(url: str = ROAD_DATA_URL) -> tuple[np.ndarray, np.ndarray]:
    """Return the bumpy-road profile (x, h).

    Attempts to download/read a data file first; if unavailable, falls back to
    the analytic synthetic profile. The expected file layout (Langtangen) is a
    text array whose row 0 holds the x-coordinates and subsequent rows hold
    road heights; the first height row is used here.

    Returns
    -------
    x : 1-D array of horizontal positions [m].
    h : 1-D array of road heights h(x) [m].
    """
    cache = FIG_DIR.parent / "bumpy_road.npy"
    try:
        if not cache.exists():
            import urllib.request
            urllib.request.urlretrieve(url, cache)
        data = np.load(cache, allow_pickle=True)
        data = np.asarray(data)
        if data.ndim == 2 and data.shape[0] >= 2:       # rows: x, h1, h2, ...
            x, h = data[0, :], data[1, :]
        elif data.ndim == 2 and data.shape[1] == 2:     # columns: (x, h)
            x, h = data[:, 0], data[:, 1]
        else:
            x = np.arange(data.size) * DX_DEFAULT
            h = data.ravel()
        return x, h
    except Exception as exc:
        print(f"Road data unavailable ({exc.__class__.__name__}); "
              f"using synthetic analytic profile.")
        return synthetic_road()


def assemble_forcing(
    x: np.ndarray,
    h: np.ndarray,
    m: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time array and forcing F(t) = -m h''(vt) v^2.

    The vehicle body feels the road *curvature* h'', not its height; the factor
    v^2 comes from the chain rule d^2/dt^2 h(vt) = h''(vt) v^2. Space and time
    are coupled through x = vt, so dt = dx/v.

    Parameters
    ----------
    x, h : road profile (uniformly spaced in x).
    m    : vehicle mass [kg].
    v    : vehicle speed [m/s].

    Returns
    -------
    t : time array (dt = dx / v).
    F : forcing array F[n] = -m * h''(x_n) * v^2.
    """
    dx = x[1] - x[0]
    d2h = np.zeros_like(h)
    d2h[1:-1] = (h[:-2] - 2.0 * h[1:-1] + h[2:]) / dx ** 2
    d2h[0] = d2h[1]        # extrapolate endpoints from first interior value
    d2h[-1] = d2h[-2]
    t = x / v
    F = -m * d2h * v ** 2
    return t, F


# ---------------------------------------------------------------------------
# Part (c): Three-panel simulation figure
# ---------------------------------------------------------------------------

def plot_three_panel(
    x: np.ndarray,
    h: np.ndarray,
    t: np.ndarray,
    F: np.ndarray,
    u: np.ndarray,
    label: str = "",
) -> None:
    """Produce a three-panel figure: road profile / forcing / displacement."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

    axes[0].plot(x, h * 1e3, color=PESC_BLUE)
    axes[0].set_ylabel("$h(x)$ [mm]")
    axes[0].set_xlabel("$x$ [m]")
    axes[0].set_title("Road Profile", fontweight="bold", color=PESC_BLUE)

    axes[1].plot(t, F, color=PESC_ORANGE)
    axes[1].set_ylabel("$F(t)$ [N]")
    axes[1].set_xlabel("$t$ [s]")
    axes[1].set_title("Forcing", fontweight="bold", color=PESC_BLUE)

    axes[2].plot(t, u * 1e3, color=PESC_BLUE)
    axes[2].set_ylabel("$u(t)$ [mm]")
    axes[2].set_xlabel("$t$ [s]")
    axes[2].set_title("Displacement", fontweight="bold", color=PESC_BLUE)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fname = f"bumpy_road_{label}.pdf" if label else "bumpy_road_default.pdf"
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight", transparent=True)
    print(f"Saved {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part (d): Parameter sweep
# ---------------------------------------------------------------------------

def parameter_sweep(x: np.ndarray, h: np.ndarray) -> None:
    """Explore different velocities and damping coefficients.

    Velocity sweep (v = 5, 10, 20 m/s) at default damping, and damping sweep
    (b = 40, 80, 200 Ns/m) at default velocity. Saves one labelled two-panel
    figure and prints peak displacements.
    """
    omega = np.sqrt(K_DEFAULT / M_DEFAULT)

    velocities = [5.0, 10.0, 20.0]
    dampings = [40.0, 80.0, 200.0]

    fig, (ax_v, ax_b) = plt.subplots(2, 1, figsize=(12, 8))

    # --- velocity sweep: plot u versus distance x (comparable across v) ---
    print("\nVelocity sweep (b = %.0f Ns/m):" % B_DEFAULT)
    for v in velocities:
        t, F = assemble_forcing(x, h, M_DEFAULT, v)
        u = solve_oscillator(t, F, M_DEFAULT, B_DEFAULT, K_DEFAULT,
                             U0_DEFAULT, V0_DEFAULT)
        peak = np.abs(u).max() * 1e3
        print(f"  v = {v:5.1f} m/s : peak |u| = {peak:6.2f} mm, "
              f"peak |F| = {np.abs(F).max():6.1f} N")
        ax_v.plot(x, u * 1e3, label=f"$v={v:.0f}$ m/s  (peak $|u|={peak:.1f}$ mm)")
    ax_v.set_xlabel("$x$ [m]")
    ax_v.set_ylabel("$u$ [mm]")
    ax_v.set_title("Velocity sweep (displacement vs distance)",
                   fontweight="bold", color=PESC_BLUE)
    ax_v.legend(fontsize=9)
    ax_v.grid(True, alpha=0.3)

    # --- damping sweep: v fixed, plot u versus time t ---
    print("\nDamping sweep (v = %.0f m/s):" % V_DEFAULT)
    for b in dampings:
        t, F = assemble_forcing(x, h, M_DEFAULT, V_DEFAULT)
        u = solve_oscillator(t, F, M_DEFAULT, b, K_DEFAULT,
                             U0_DEFAULT, V0_DEFAULT)
        zeta = b / (2.0 * M_DEFAULT * omega)
        peak = np.abs(u).max() * 1e3
        print(f"  b = {b:5.1f} Ns/m: zeta = {zeta:.2f}, peak |u| = {peak:6.2f} mm")
        ax_b.plot(t, u * 1e3,
                  label=fr"$b={b:.0f}$  ($\zeta={zeta:.2f}$, peak $|u|={peak:.1f}$ mm)")
    ax_b.set_xlabel("$t$ [s]")
    ax_b.set_ylabel("$u$ [mm]")
    ax_b.set_title("Damping sweep (displacement vs time)",
                   fontweight="bold", color=PESC_BLUE)
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "bumpy_sweep.pdf", dpi=300,
                bbox_inches="tight", transparent=True)
    print("Saved bumpy_sweep.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_solver()

    x, h = load_road_data()
    t, F = assemble_forcing(x, h, m=M_DEFAULT, v=V_DEFAULT)
    u = solve_oscillator(t, F, M_DEFAULT, B_DEFAULT, K_DEFAULT, U0_DEFAULT, V0_DEFAULT)
    plot_three_panel(x, h, t, F, u, label="default")
    parameter_sweep(x, h)
