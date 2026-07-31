import numpy as np

def model1_ode(t, y, r_S, r_R, d_max_S, d_max_R, EC50_S, EC50_R, mu, A):
    S, R = y
    dSdt = r_S * S - (d_max_S * (A / EC50_S)) * S
    dRdt = r_R * R + mu * r_S * S - (d_max_R * (A / EC50_R)) * R
    return [dSdt, dRdt]

def model2_ode(t, y, r_S, r_R, K, mu, E_max_S, E_max_R, EC50_S, EC50_R, H, A):
    S, R = y
    N = S + R
    growth_factor = 1.0 - (N / K) if K > 0 else 1.0

    phi_S = (E_max_S * (A**H)) / ((EC50_S**H) + (A**H)) if A > 0 else 0
    phi_R = (E_max_R * (A**H)) / ((EC50_R**H) + (A**H)) if A > 0 else 0

    dSdt = r_S * S * growth_factor * (1.0 - mu) - phi_S * S
    dRdt = r_R * R * growth_factor + mu * r_S * S * growth_factor - phi_R * R
    return [dSdt, dRdt]

def model3_ode(t, y, r_S, r_R, K, mu, E_max_S, E_max_R, EC50_S, EC50_R, H, k_d):
    S, R, A = y
    N = S + R
    growth_factor = 1.0 - (N / K) if K > 0 else 1.0

    phi_S = (E_max_S * (A**H)) / ((EC50_S**H) + (A**H)) if A > 0 else 0
    phi_R = (E_max_R * (A**H)) / ((EC50_R**H) + (A**H)) if A > 0 else 0

    dSdt = r_S * S * growth_factor * (1.0 - mu) - phi_S * S
    dRdt = r_R * R * growth_factor + mu * r_S * S * growth_factor - phi_R * R
    dAdt = -k_d * A  

    return [dSdt, dRdt, dAdt]