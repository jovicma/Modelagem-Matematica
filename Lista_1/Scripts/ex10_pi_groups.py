"""
Exercise 10 — Dimensional Matrix Tool and Scaling Verification
"""

import numpy as np
from scipy.linalg import null_space
from fractions import Fraction
import math

# ---------------------------------------------------------------------------
# Test cases (DO NOT MODIFY)
# ---------------------------------------------------------------------------

# (i) Marble on a cone: variables [Delta_t, m, R, g]
#     Dimensions: M, L, T
MARBLE_D = np.array([
    # Dt   m   R   g
    [ 0,   1,  0,  0],   # M
    [ 0,   0,  1,  1],   # L
    [ 1,   0,  0, -2],   # T
], dtype=float)
MARBLE_VARS = ["Delta_t", "m", "R", "g"]
MARBLE_DIMS = ["M", "L", "T"]

# (ii) Ideal gas pressure: variables [p, N/V, m, v_rms]
#      Dimensions: M, L, T
GAS_D = np.array([
    # p    N/V   m    v
    [ 1,   0,   1,   0],   # M
    [-1,  -3,   0,   1],   # L
    [-2,   0,   0,  -1],   # T
], dtype=float)
GAS_VARS = ["p", "N_over_V", "m", "v_rms"]
GAS_DIMS = ["M", "L", "T"]

# (iii) Damped forced oscillator: [u, t, m, b, k, A, psi, u_0, v_0]
#       Dimensions: M, L, T
OSCILLATOR_D = np.array([
    # u   t   m   b    k    A   psi  u0  v0
    [ 0,  0,  1,  1,   1,   1,   0,  0,  0],   # M
    [ 1,  0,  0,  0,   0,   1,   0,  1,  1],   # L
    [ 0,  1,  0, -1,  -2,  -2,  -1,  0, -1],   # T
], dtype=float)
OSCILLATOR_VARS = ["u", "t", "m", "b", "k", "A", "psi", "u_0", "v_0"]
OSCILLATOR_DIMS = ["M", "L", "T"]

# (c) New System: Capillary Rise (Ascensao Capilar)
#     Variables: h (height), gamma (surface tension), rho (density), g (gravity), r (radius)
#     Dimensions: M, L, T
CAPILLARY_D = np.array([
    # h  gamma rho   g   r
    [ 0,   1,   1,   0,  0],  # M
    [ 1,   0,  -3,   1,  1],  # L
    [ 0,  -2,   0,  -2,  0],  # T
], dtype=float)
CAPILLARY_VARS = ["h", "gamma", "rho", "g", "r"]
CAPILLARY_DIMS = ["M", "L", "T"]


def find_pi_groups(D, variable_names, dimension_names):
    """Compute independent dimensionless products from a dimensional matrix."""
    
    rank = np.linalg.matrix_rank(D)
    n_vars = D.shape[1]
    n_pi = n_vars - rank
    
    # Get the null space
    ns = null_space(D)
    
    pi_groups = []
    
    # Process each basis vector of the null space
    for i in range(n_pi):
        v = ns[:, i]
        
        # Rationalize the vector (convert to simple integers)
        abs_v = np.abs(v)
        tol = 1e-7
        if np.any(abs_v > tol):
            min_nonzero = np.min(abs_v[abs_v > tol])
            v_scaled = v / min_nonzero
        else:
            v_scaled = v
            
        fracs = [Fraction(x).limit_denominator(1000) for x in v_scaled]
        
        # Find Least Common Multiple (LCM) of denominators
        lcm = 1
        for f in fracs:
            lcm = abs(lcm * f.denominator) // math.gcd(lcm, f.denominator)
            
        ints = [int(round(f * lcm)) for f in fracs]
        
        # Ensure the first non-zero exponent is positive for aesthetic consistency
        for val in ints:
            if val != 0:
                if val < 0:
                    ints = [-x for x in ints]
                break
                
        # Format the output string
        terms = []
        for var_name, power in zip(variable_names, ints):
            if power == 0:
                continue
            elif power == 1:
                terms.append(var_name)
            else:
                terms.append(f"{var_name}^{power}")
                
        pi_str = f"pi_{i+1} = " + " * ".join(terms)
        pi_groups.append(pi_str)
        
    return rank, n_pi, pi_groups


def main():
    print("=" * 60)
    print("(i) Marble on a cone")
    print("=" * 60)
    rank, n_pi, groups = find_pi_groups(MARBLE_D, MARBLE_VARS, MARBLE_DIMS)
    print(f"  Rank = {rank}, N_P = {n_pi}")
    for g in groups:
        print(f"  {g}")

    print()
    print("=" * 60)
    print("(ii) Ideal gas pressure")
    print("=" * 60)
    rank, n_pi, groups = find_pi_groups(GAS_D, GAS_VARS, GAS_DIMS)
    print(f"  Rank = {rank}, N_P = {n_pi}")
    for g in groups:
        print(f"  {g}")

    print()
    print("=" * 60)
    print("(iii) Damped forced oscillator")
    print("=" * 60)
    rank, n_pi, groups = find_pi_groups(OSCILLATOR_D, OSCILLATOR_VARS, OSCILLATOR_DIMS)
    print(f"  Rank = {rank}, N_P = {n_pi}")
    for g in groups:
        print(f"  {g}")
    print(f"\n  Compare: scaling the ODE gives 4 numbers (zeta, delta, gamma, beta)")
    print(f"  Discrepancy: pi theorem gives {n_pi}, scaling gives 4")
    
    print()
    print("=" * 60)
    print("(c) New System: Capillary Rise (Ascensao Capilar)")
    print("=" * 60)
    rank, n_pi, groups = find_pi_groups(CAPILLARY_D, CAPILLARY_VARS, CAPILLARY_DIMS)
    print(f"  Rank = {rank}, N_P = {n_pi}")
    for g in groups:
        print(f"  {g}")


if __name__ == "__main__":
    main()