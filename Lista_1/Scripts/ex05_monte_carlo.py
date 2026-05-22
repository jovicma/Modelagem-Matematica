import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------
P_VAL = 101.3e3      # Pressure [Pa]
P_STD = 0.5e3        # Pressure uncertainty [Pa]
V_VAL = 2.50e-3      # Volume [m^3] (2.50 L)
V_STD = 0.05e-3      # Volume uncertainty [m^3]
T_VAL = 295.0        # Temperature [K]
T_STD = 2.0          # Temperature uncertainty [K]
R_GAS = 8.314        # Gas constant [J/(mol·K)] (exact)

FIG_DIR = Path(__file__).resolve().parent / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_n(p, V, T):
    """Compute number of moles from ideal gas law: n = pV/(RT)."""
    return p * V / (R_GAS * T)


def analytical_propagation():
    """(a) Compute n and its uncertainty using the derivative rule."""
    n_val = compute_n(P_VAL, V_VAL, T_VAL)
    
    # Relative uncertainty in quadrature for products/quotients
    rel_err_sq = (P_STD / P_VAL)**2 + (V_STD / V_VAL)**2 + (T_STD / T_VAL)**2
    n_std = n_val * np.sqrt(rel_err_sq)
    
    return n_val, n_std


def monte_carlo_propagation(N=50_000, seed=42):
    """(b) Monte Carlo propagation with independent p, V, T."""
    rng = np.random.default_rng(seed)
    
    p_samples = rng.normal(P_VAL, P_STD, N)
    V_samples = rng.normal(V_VAL, V_STD, N)
    T_samples = rng.normal(T_VAL, T_STD, N)
    
    n_samples = compute_n(p_samples, V_samples, T_samples)
    return n_samples


def monte_carlo_correlated(N=50_000, rho_pT=0.3, seed=42):
    """(e) Monte Carlo with correlated (p, T)."""
    rng = np.random.default_rng(seed)
    
    # Covariance matrix for [p, T]
    cov_pT = rho_pT * P_STD * T_STD
    cov_matrix = [
        [P_STD**2, cov_pT],
        [cov_pT,   T_STD**2]
    ]
    
    # Sample (p, T) together
    pT_samples = rng.multivariate_normal([P_VAL, T_VAL], cov_matrix, N)
    p_samples = pT_samples[:, 0]
    T_samples = pT_samples[:, 1]
    
    # V is still independent
    V_samples = rng.normal(V_VAL, V_STD, N)
    
    n_samples = compute_n(p_samples, V_samples, T_samples)
    return n_samples


def plot_results(n_samples, n_analytical, n_std_analytical, label="independent"):
    """(c) Plot histogram and Q-Q plot of Monte Carlo samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Histogram
    ax1.hist(n_samples, bins=100, density=True, alpha=0.7, color='steelblue', label='MC Samples')
    
    # Plot analytical normal distribution overlay for comparison
    x = np.linspace(n_analytical - 4*n_std_analytical, n_analytical + 4*n_std_analytical, 200)
    y = stats.norm.pdf(x, n_analytical, n_std_analytical)
    ax1.plot(x, y, 'r--', lw=2, label='Analytical Dist')
    
    ax1.axvline(n_analytical, color='k', linestyle='dashed', linewidth=1.5, label='Analytical Mean')
    ax1.set_xlabel('Number of moles (n)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'Distribution of n ({label})')
    ax1.legend()
    
    # Right panel: Q-Q Plot
    stats.probplot(n_samples, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot ({label})')
    
    plt.tight_layout()
    
    # Save figures
    filepath_pdf = FIG_DIR / f"ex05_{label}.pdf"
    filepath_png = FIG_DIR / f"ex05_{label}.png"
    plt.savefig(filepath_pdf)
    plt.savefig(filepath_png)
    plt.close()


def main():
    # (a) Analytical
    n_val, n_std = analytical_propagation()
    print(f"Analytical: n = {n_val:.6f} +/- {n_std:.6f} mol")

    # (b) Monte Carlo (independent)
    n_mc = monte_carlo_propagation()
    print(f"MC (indep): n = {n_mc.mean():.6f} +/- {n_mc.std():.6f} mol")

    # (c) Plots
    plot_results(n_mc, n_val, n_std, label="independent")

    # (e) Monte Carlo (correlated)
    n_mc_corr = monte_carlo_correlated()
    print(f"MC (corr):  n = {n_mc_corr.mean():.6f} +/- {n_mc_corr.std():.6f} mol")
    plot_results(n_mc_corr, n_val, n_std, label="correlated")


if __name__ == "__main__":
    main()