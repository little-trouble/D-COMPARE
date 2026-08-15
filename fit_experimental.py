import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from models import model1_ode, model2_ode, model3_ode
from metrics import calculate_r2, calculate_rmse, calculate_aic, calculate_tres

df = pd.read_csv("experimental_timekill_ecoli.csv")

df["cfu_ml"] = 10 ** df["log10_cfu_ml"]

drugs = df["drug"].unique()

# Fixed parameters
K = 1e9
mu = 1e-8
H_default = {
    "ciprofloxacin": 2.0,
    "ampicillin": 1.0
}

R0 = 10.0

def fit_model1(data):

    H = H_default[data["drug"].iloc[0]]

    def objective(params):

        r_S, r_R, d_max_S, EC50_S = params
        error = 0

        for concentration in data["concentration"].unique():

            sub = data[data["concentration"] == concentration]

            t = sub["time_min"].values / 60
            y_obs = sub["log10_cfu_ml"].values

            S0 = 10 ** sub["log10_cfu_ml"].iloc[0]

            sol = solve_ivp(
                lambda t, y: model1_ode(
                    t, y,
                    r_S, r_R,
                    d_max_S,
                    d_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    mu,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

            if not sol.success:
                return 1e10

            N_pred = sol.y[0] + sol.y[1]
            log_pred = np.log10(np.maximum(N_pred, 1))

            error += np.sum((y_obs - log_pred) ** 2)

        return error

    result = minimize(
        objective,
        [0.8, 0.6, 2.0, 0.5],
        bounds=[
            (0.01, 2.5),
            (0.01, 2.5),
            (0.01, 10),
            (1e-5, max(data["concentration"].max(), 1))
        ],
        method="L-BFGS-B"
    )

    return result.x

def fit_model2(data):

    H = H_default[data["drug"].iloc[0]]

    def objective(params):

        r_S, r_R, E_max_S, EC50_S = params
        error = 0

        for concentration in data["concentration"].unique():

            sub = data[data["concentration"] == concentration]

            t = sub["time_min"].values / 60
            y_obs = sub["log10_cfu_ml"].values

            S0 = 10 ** sub["log10_cfu_ml"].iloc[0]

            sol = solve_ivp(
                lambda t, y: model2_ode(
                    t, y,
                    r_S, r_R,
                    K, mu,
                    E_max_S,
                    E_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    H,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

            if not sol.success:
                return 1e10

            N_pred = sol.y[0] + sol.y[1]
            log_pred = np.log10(np.maximum(N_pred, 1))

            error += np.sum((y_obs - log_pred) ** 2)

        return error

    result = minimize(
        objective,
        [0.8, 0.6, 2.0, 0.5],
        bounds=[
            (0.01, 2.5),
            (0.01, 2.5),
            (0.01, 10),
            (1e-5, max(data["concentration"].max(), 1))
        ],
        method="L-BFGS-B"
    )

    return result.x


def fit_model3(data):

    H = H_default[data["drug"].iloc[0]]

    def objective(params):

        r_S, r_R, E_max_S, EC50_S, k_d = params
        error = 0

        for concentration in data["concentration"].unique():

            sub = data[data["concentration"] == concentration]

            t = sub["time_min"].values / 60
            y_obs = sub["log10_cfu_ml"].values

            S0 = 10 ** sub["log10_cfu_ml"].iloc[0]

            sol = solve_ivp(
                lambda t, y: model3_ode(
                    t, y,
                    r_S, r_R,
                    K, mu,
                    E_max_S,
                    E_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    H,
                    k_d
                ),
                (t[0], t[-1]),
                [S0, R0, concentration],
                t_eval=t
            )

            if not sol.success:
                return 1e10

            N_pred = sol.y[0] + sol.y[1]
            log_pred = np.log10(np.maximum(N_pred, 1))

            error += np.sum((y_obs - log_pred) ** 2)

        return error

    result = minimize(
        objective,
        [0.8, 0.6, 2.0, 0.5, 0.05],
        bounds=[
            (0.01, 2.5),
            (0.01, 2.5),
            (0.01, 10),
            (1e-5, max(data["concentration"].max(), 1)),
            (0.0001, 1.0)
        ],
        method="L-BFGS-B"
    )

    return result.x


def evaluate_model(data, model, params):

    H = H_default[data["drug"].iloc[0]]

    observed = []
    predicted = []

    for concentration in data["concentration"].unique():

        sub = data[data["concentration"] == concentration]

        t = sub["time_min"].values / 60
        y_obs = sub["log10_cfu_ml"].values

        S0 = 10 ** sub["log10_cfu_ml"].iloc[0]

        if model == 1:

            r_S, r_R, d_max_S, EC50_S = params

            sol = solve_ivp(
                lambda t, y: model1_ode(
                    t, y,
                    r_S, r_R,
                    d_max_S,
                    d_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    mu,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

        elif model == 2:

            r_S, r_R, E_max_S, EC50_S = params

            sol = solve_ivp(
                lambda t, y: model2_ode(
                    t, y,
                    r_S, r_R,
                    K, mu,
                    E_max_S,
                    E_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    H,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

        else:

            r_S, r_R, E_max_S, EC50_S, k_d = params

            sol = solve_ivp(
                lambda t, y: model3_ode(
                    t, y,
                    r_S, r_R,
                    K, mu,
                    E_max_S,
                    E_max_S * 0.1,
                    EC50_S,
                    EC50_S * 10,
                    H,
                    k_d
                ),
                (t[0], t[-1]),
                [S0, R0, concentration],
                t_eval=t
            )

        N_pred = sol.y[0] + sol.y[1]

        observed.extend(y_obs)
        predicted.extend(np.log10(np.maximum(N_pred, 1)))

    observed = np.array(observed)
    predicted = np.array(predicted)

    return (
        calculate_r2(observed, predicted),
        calculate_rmse(observed, predicted),
        calculate_aic(observed, predicted, len(params))
    )
for drug in drugs:

    data = df[df["drug"] == drug].copy()

    print(drug.upper())
    
    p1 = fit_model1(data)
    p2 = fit_model2(data)
    p3 = fit_model3(data)

    r2_1, rmse_1, aic_1 = evaluate_model(data, 1, p1)
    r2_2, rmse_2, aic_2 = evaluate_model(data, 2, p2)
    r2_3, rmse_3, aic_3 = evaluate_model(data, 3, p3)

    print("\nModel 1")
    print("parameters =", p1)
    print("R2 =", r2_1)
    print("RMSE =", rmse_1)
    print("AIC =", aic_1)

    print("\nModel 2")
    print("parameters =", p2)
    print("R2 =", r2_2)
    print("RMSE =", rmse_2)
    print("AIC =", aic_2)

    print("\nModel 3")
    print("parameters =", p3)
    print("R2 =", r2_3)
    print("RMSE =", rmse_3)
    print("AIC =", aic_3)

