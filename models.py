import numpy as np

def logistic_factor(S, R, K):
    N = S + R
    if K <= 0:
        return 1.0
    return max(0.0, 1.0 - N / K)

def model1_ode(t,y, r_S,r_R,K,mu,alpha_S,alpha_R, A):
    S, R = y
    growth = logistic_factor(S, R, K)

    dSdt = ( r_S * S * growth * (1.0 - mu) - alpha_S * A * S)
    dRdt = ( r_R * R * growth  + mu * r_S * S * growth - alpha_R * A * R )
    return [dSdt, dRdt]

def hill_effect( A,  E_max,EC50, H):
    if A <= 0:
        return 0.0

    numerator = E_max * (A ** H)
    denominator = (EC50 ** H) + (A ** H)

    if denominator == 0:
        return 0.0
    return numerator / denominator

def model2_ode( t, y, r_S,  r_R,K, mu, Emax_S, Emax_R, EC50_S, EC50_R, H,  A):
    S, R = y
    growth = logistic_factor(S, R, K)

    phi_S = hill_effect( A, Emax_S, EC50_S, H )
    phi_R = hill_effect( A, Emax_R, EC50_R, H )

    dSdt = (r_S * S * growth * (1.0 - mu)- phi_S * S)
    dRdt = ( r_R * R * growth + mu * r_S * S * growth- phi_R * R )
    return [dSdt, dRdt]

def model3_ode(t, y, r_S, r_R, K, mu, Emax_S, Emax_R, EC50_S, EC50_R, H, k_d):
    S, R, A = y
    growth = logistic_factor(S, R, K)

    phi_S = hill_effect( A, Emax_S, EC50_S, H )
    phi_R = hill_effect( A, Emax_R, EC50_R, H )

    dSdt = ( r_S * S * growth * (1.0 - mu) - phi_S * S )
    dRdt = ( r_R * R * growth + mu * r_S * S * growth- phi_R * R )
    dAdt = -k_d * A
    return [dSdt, dRdt, dAdt ]

def get_total_population(solution):
    return solution.y[0] + solution.y[1]

def get_resistance_ratio(solution):
    total = solution.y[0] + solution.y[1]
    total = np.where(total == 0, 1e-12, total)
    return solution.y[1] / total