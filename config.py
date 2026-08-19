import pandas as pd
import os

drug = "ciprofloxacin"   # or "ampicillin"

S0 = 1e5
R0 = 10
A0 = 1.0
K = 1e9
mu = 1e-8

CSV_PATH = os.path.join(os.path.dirname(__file__), "model_parameters_config_aligned.csv")

_df = pd.read_csv(CSV_PATH)
_df = _df[_df["drug"] == drug].set_index("parameter")["value"]

def _get(name, cast=float):
    """Pull a single parameter value for the current drug and cast it."""
    return cast(_df.loc[name])

S0 = _get("S0")
R0 = _get("R0")
A0 = _get("A0")

K  = _get("K")
mu = _get("mu")
H  = _get("H")
k_d = _get("k_d")

t_start = 0
t_end = 48
n_points = 500

initial_guess = [
    _get("r_S_initial_guess"),     # r_S
    _get("r_R_initial_guess"),     # r_R
    _get("Emax_S_initial_guess"),  # Emax_S
    _get("EC50_S_initial_guess"),  # EC50_S
]

bounds = [
    (_get("r_S_bounds_low"),    _get("r_S_bounds_high")),
    (_get("r_R_bounds_low"),    _get("r_R_bounds_high")),
    (_get("Emax_S_bounds_low"), _get("Emax_S_bounds_high")),
    (_get("EC50_S_bounds_low"), _get("EC50_S_bounds_high")),
]

if __name__ == "__main__":
    print(f"drug = {drug}")
    print(f"S0={S0}, R0={R0}, A0={A0}, K={K}, mu={mu}, H={H}, k_d={k_d}")
    print(f"initial_guess = {initial_guess}")
    print(f"bounds = {bounds}")