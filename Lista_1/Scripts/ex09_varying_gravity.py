import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_EARTH = 6.371e6     # Earth radius [m]
G_SURFACE = 9.81      # Surface gravity [m/s^2]

# Salvando as figuras na mesma pasta do script
FIG_DIR = Path(__file__).resolve().parent / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Epsilon^2 values to explore
EPSILON_SQ = [0.01, 0.1, 0.5, 1.0, 2.0]


def solve_scaled_projectile(epsilon_sq, T_bar_max=100.0):
    """(c) Solve the scaled projectile equation (Primary Scaling)."""
    
    def odefun(t, y):
        h_bar, v_bar = y
        return [v_bar, -1.0 / (epsilon_sq * (h_bar + 1.0)**2)]

    def hit_ground(t, y):
        return y[0] # Para a integração quando h_bar cruzar 0
    hit_ground.terminal = True
    hit_ground.direction = -1

    # Condições iniciais: h_bar(0) = 0, h_bar'(0) = 1
    # max_step controla a suavidade da curva, especialmente para epsilon grandes
    sol = solve_ivp(odefun, [0, T_bar_max], [0.0, 1.0], events=hit_ground, max_step=0.05)
    return sol.t, sol.y[0]


def solve_alternative_scaling(epsilon_sq, T_bar_max=100.0):
    """(d) Solve with alternative scaling h_c = v_0^2/g, t_c = v_0/g."""
    
    def odefun(t, y):
        h_bar, v_bar = y
        return [v_bar, -1.0 / (1.0 + epsilon_sq * h_bar)**2]

    def hit_ground(t, y):
        return y[0]
    hit_ground.terminal = True
    hit_ground.direction = -1

    sol = solve_ivp(odefun, [0, T_bar_max], [0.0, 1.0], events=hit_ground, max_step=0.05)
    return sol.t, sol.y[0]


def plot_trajectories(results, title, filename):
    """Plot h_bar vs t_bar for multiple epsilon^2 values."""
    plt.figure(figsize=(9, 6))
    
    for t_bar, h_bar, eps_sq in results:
        # Se eps_sq == 2.0, o projétil atinge velocidade de escape e não cai
        label_str = rf"$\epsilon^2 = {eps_sq}$"
        if eps_sq >= 2.0:
            label_str += " (Escape)"
            
        plt.plot(t_bar, h_bar, label=label_str)

    plt.title(title, fontsize=14)
    plt.xlabel(r"Tempo Adimensional $\bar{t}$", fontsize=12)
    plt.ylabel(r"Altura Adimensional $\bar{h}$", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(FIG_DIR / f"{filename}.pdf")
    plt.savefig(FIG_DIR / f"{filename}.png")
    plt.close()


def main():
    print("Simulando Escalamento Primario...")
    results_primary = []
    for eps_sq in EPSILON_SQ:
        t_bar, h_bar = solve_scaled_projectile(eps_sq)
        results_primary.append((t_bar, h_bar, eps_sq))
        print(f"  eps^2 = {eps_sq}: max h_bar = {h_bar.max():.3f}")

    plot_trajectories(results_primary,
                      r"Escalamento Primário ($h_c = R$)",
                      "ex09_primary_scaling")

    print("\nSimulando Escalamento Alternativo...")
    results_alt = []
    for eps_sq in EPSILON_SQ:
        t_bar, h_bar = solve_alternative_scaling(eps_sq)
        results_alt.append((t_bar, h_bar, eps_sq))
        print(f"  eps^2 = {eps_sq}: max h_bar = {h_bar.max():.3f}")

    plot_trajectories(results_alt,
                      r"Escalamento Alternativo ($h_c = v_0^2/g$)",
                      "ex09_alternative_scaling")
    
    print("\nGraficos salvos na pasta 'Figures'.")


if __name__ == "__main__":
    main()