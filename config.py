drug = "ciprofloxacin"     

S0 = 1e5
R0 = 10
A0 = 1.0

K = 1e9
mu = 1e-8

if drug == "ciprofloxacin":
    H = 2.0
else:
    H = 1.0
        
k_d = 0.05
    
t_start = 0
t_end = 48
n_points = 500
     
initial_guess = [
    0.8,   # r_S
    0.6,   # r_R
    2.0,   # Emax_S
    0.5    # EC50_S
]

bounds = [
    (0.1, 2.0),      # r_S
    (0.1, 2.0),      # r_R
    (0.1, 10.0),     # Emax_S
    (0.01, 5.0)      # EC50_S
]