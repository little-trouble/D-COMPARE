import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from config import *
from models import model1_ode, model2_ode, model3_ode
from metrics import compare_models, tres

t_data = np.array([0, 2, 4, 6, 8, 12, 16, 24, 32, 48])



log_data = np.log10(N_data + 1)

y0_m1_m2 = [S0, R0]
y0_m3 = [S0, R0, A0]

t_eval = np.linspace(t_start, t_end, n_points)

def objective(params):  
    r_S, r_R, Emax_S, EC50_S = params
    sol = solve_ivp(
        lambda t, y:model2_ode(t, y, r_S, r_R, K, mu, Emax_S, Emax_S * 0.1, EC50_S, EC50_S * 10, H, A0 ),
        (t_start, t_end),
        y0_m1_m2,
        t_eval=t_data
    )

    pred = sol.y[0] + sol.y[1]
    log_pred = np.log10(pred + 1)
    return np.sum((log_data - log_pred) ** 2)

result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

r_S, r_R, Emax_S, EC50_S = result.x
Emax_R = Emax_S * 0.1
EC50_R = EC50_S * 10

print("Best Parameters ")
print(f"r_S     = {r_S:.4f}")
print(f"r_R     = {r_R:.4f}")
print(f"Emax_S  = {Emax_S:.4f}")
print(f"EC50_S  = {EC50_S:.4f}")

sol1 = solve_ivp(lambda t, y:
    model1_ode( t, y, r_S, r_R, K, mu, 0.8, 0.1, A0),
    (t_start, t_end),
    y0_m1_m2,
    t_eval=t_eval
)

sol2 = solve_ivp(lambda t, y:
    model2_ode(t, y, r_S, r_R, K, mu, Emax_S, Emax_R, EC50_S, EC50_R, H, A0 ),
    (t_start, t_end),
    y0_m1_m2,
    t_eval=t_eval
)

sol3 = solve_ivp(lambda t, y:
    model3_ode( t, y, r_S, r_R, K, mu, Emax_S, Emax_R, EC50_S, EC50_R, H, k_d ),
    (t_start, t_end),
    y0_m3,
    t_eval=t_eval
)

pred1 = np.interp(t_data, sol1.t, sol1.y[0] + sol1.y[1])
pred2 = np.interp(t_data, sol2.t, sol2.y[0] + sol2.y[1])
pred3 = np.interp(t_data, sol3.t, sol3.y[0] + sol3.y[1])

log_pred1 = np.log10(pred1 + 1)
log_pred2 = np.log10(pred2 + 1)
log_pred3 = np.log10(pred3 + 1)

print("Model Comparison")
compare_models("Model 1", log_data, log_pred1,4)
compare_models("Model 2", log_data, log_pred2,4)
compare_models("Model 3", log_data, log_pred3, 5)
                    
tres1 = tres(sol1.t, sol1.y[0], sol1.y[1])
tres2 = tres(sol2.t, sol2.y[0], sol2.y[1])
tres3 = tres(sol3.t, sol3.y[0], sol3.y[1])

print("Time to Resistance")

print(f"Model 1 : {tres1:.2f} hr")
print(f"Model 2 : {tres2:.2f} hr")
print(f"Model 3 : {tres3:.2f} hr")

print("Sensitivity Analysis")

change = 0.20

parameters = {
    "r_S": r_S,
    "r_R": r_R,
    "Emax_S": Emax_S,
    "EC50_S": EC50_S
}

for name, value in parameters.items():
    plus = value * (1 + change)
    minus = value * (1 - change)

    print(
        f"{name:8s}  "
        f"-20% = {minus:.4f}   "
        f"+20% = {plus:.4f}"
    )
    
plt.figure(figsize=(9,6))

plt.scatter(
        t_data,
        N_data,
        color="black",
        label="Observed Data",
        zorder=10                       
)

plt.plot(
    sol1.t,
    sol1.y[0] + sol1.y[1],
    "--",
    linewidth=2,
    label="Model 1"
)

plt.plot(
    sol2.t,
    sol2.y[0] + sol2.y[1],
    linewidth=2,
    label="Model 2"

)

plt.plot(
    sol3.t,
    sol3.y[0] + sol3.y[1],
    "-.",
    linewidth=2,
    label="Model 3"
)

plt.yscale("log")

plt.xlabel("Time (hours)")
plt.ylabel("Population (cells/mL)")
plt.title("Comparison of Mathematical Models")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "model_comparison.png",
        dpi=300
)
plt.show()

plt.figure(figsize=(9,6))
ratio1 = sol1.y[1] / (sol1.y[0] + sol1.y[1])
ratio2 = sol2.y[1] / (sol2.y[0] + sol2.y[1])
ratio3 = sol3.y[1] / (sol3.y[0] + sol3.y[1])

plt.plot(sol1.t,ratio1,"--",label="Model 1")
plt.plot(sol2.t,ratio2,label="Model 2")
plt.plot(sol3.t,ratio3,"-.",label="Model 3")

plt.axhline(
        0.5,
        color="red",
        linestyle=":"
)

plt.xlabel("Time (hours)")
plt.ylabel("R / (S + R)")

plt.title("Resistance Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    "resistance_ratio.png",
        dpi=300
        )

plt.show()
print("Finish")
