import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from config import K, mu, R0, k_d, initial_guess, bounds
from models import model1_ode, model2_ode, model3_ode
from metrics import compare_models, tres

df = pd.read_csv("experimental_timekill_ecoli.csv")

drugs = df["drug"].unique()

H_default = {
    "ciprofloxacin": 2.0,
    "ampicillin": 1.0
}

def get_initial_S(data):
    """
    Initial susceptible population for one
    drug-concentration group.
    """
    first_row = data.sort_values("time_min").iloc[0]
    return 10 ** first_row["log10_cfu_ml"]

def fit_model1(data):

    def objective(params):

        r_S, r_R, alpha_S, alpha_R = params

        error = 0

        for concentration in data["concentration"].unique():

            sub = data[
                data["concentration"] == concentration
            ].sort_values("time_min")

            t = sub["time_min"].values / 60.0

            y_obs = sub["log10_cfu_ml"].values

            S0 = get_initial_S(sub)

            sol = solve_ivp(
                lambda t, y: model1_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    alpha_S,
                    alpha_R,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

            if not sol.success:
                return 1e10

            N_pred = sol.y[0] + sol.y[1]

            log_pred = np.log10(
                np.maximum(N_pred, 1)
            )

            error += np.sum(
                (y_obs - log_pred) ** 2
            )

        return error


    # Model 1 uses alpha instead of Emax/EC50.
    # Use Emax initial guess as a reasonable starting scale.
    alpha_initial = initial_guess[2]

    alpha_bounds = bounds[2]

    model1_initial = [
        initial_guess[0],
        initial_guess[1],
        alpha_initial,
        alpha_initial * 0.1
    ]

    model1_bounds = [
        bounds[0],
        bounds[1],
        alpha_bounds,
        alpha_bounds
    ]

    result = minimize(
        objective,
        model1_initial,
        bounds=model1_bounds,
        method="L-BFGS-B"
    )

    return result.x


def fit_model2(data):

    H = H_default[data["drug"].iloc[0]]

    def objective(params):

        r_S, r_R, Emax_S, EC50_S = params

        Emax_R = Emax_S * 0.1
        EC50_R = EC50_S * 10

        error = 0

        for concentration in data["concentration"].unique():

            sub = data[
                data["concentration"] == concentration
            ].sort_values("time_min")

            t = sub["time_min"].values / 60.0

            y_obs = sub["log10_cfu_ml"].values

            S0 = get_initial_S(sub)

            sol = solve_ivp(
                lambda t, y: model2_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    Emax_S,
                    Emax_R,
                    EC50_S,
                    EC50_R,
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

            log_pred = np.log10(
                np.maximum(N_pred, 1)
            )

            error += np.sum(
                (y_obs - log_pred) ** 2
            )

        return error


    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method="L-BFGS-B"
    )

    return result.x


def fit_model3(data):

    H = H_default[data["drug"].iloc[0]]

    def objective(params):

        r_S, r_R, Emax_S, EC50_S, kd = params

        Emax_R = Emax_S * 0.1
        EC50_R = EC50_S * 10

        error = 0

        for concentration in data["concentration"].unique():

            sub = data[
                data["concentration"] == concentration
            ].sort_values("time_min")

            t = sub["time_min"].values / 60.0

            y_obs = sub["log10_cfu_ml"].values

            S0 = get_initial_S(sub)

            sol = solve_ivp(
                lambda t, y: model3_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    Emax_S,
                    Emax_R,
                    EC50_S,
                    EC50_R,
                    H,
                    kd
                ),
                (t[0], t[-1]),
                [S0, R0, concentration],
                t_eval=t
            )

            if not sol.success:
                return 1e10

            N_pred = sol.y[0] + sol.y[1]

            log_pred = np.log10(
                np.maximum(N_pred, 1)
            )

            error += np.sum(
                (y_obs - log_pred) ** 2
            )

        return error


    model3_initial = [
        initial_guess[0],
        initial_guess[1],
        initial_guess[2],
        initial_guess[3],
        k_d
    ]

    model3_bounds = [
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        (0.0001, 1.0)
    ]


    result = minimize(
        objective,
        model3_initial,
        bounds=model3_bounds,
        method="L-BFGS-B"
    )

    return result.x

def evaluate_model(data, model, params):

    H = H_default[data["drug"].iloc[0]]

    observed = []
    predicted = []

    solutions = []

    for concentration in data["concentration"].unique():

        sub = data[
            data["concentration"] == concentration
        ].sort_values("time_min")

        t = sub["time_min"].values / 60.0

        y_obs = sub["log10_cfu_ml"].values

        S0 = get_initial_S(sub)

        if model == 1:

            r_S, r_R, alpha_S, alpha_R = params

            sol = solve_ivp(
                lambda t, y: model1_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    alpha_S,
                    alpha_R,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

        elif model == 2:

            r_S, r_R, Emax_S, EC50_S = params

            Emax_R = Emax_S * 0.1
            EC50_R = EC50_S * 10

            sol = solve_ivp(
                lambda t, y: model2_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    Emax_S,
                    Emax_R,
                    EC50_S,
                    EC50_R,
                    H,
                    concentration
                ),
                (t[0], t[-1]),
                [S0, R0],
                t_eval=t
            )

        else:

            r_S, r_R, Emax_S, EC50_S, kd = params

            Emax_R = Emax_S * 0.1
            EC50_R = EC50_S * 10

            sol = solve_ivp(
                lambda t, y: model3_ode(
                    t,
                    y,
                    r_S,
                    r_R,
                    K,
                    mu,
                    Emax_S,
                    Emax_R,
                    EC50_S,
                    EC50_R,
                    H,
                    kd
                ),
                (t[0], t[-1]),
                [S0, R0, concentration],
                t_eval=t
            )


        if not sol.success:
            continue


        N_pred = sol.y[0] + sol.y[1]

        observed.extend(y_obs)

        predicted.extend(
            np.log10(
                np.maximum(N_pred, 1)
            )
        )

        solutions.append(
            (concentration, sol)
        )


    observed = np.array(observed)

    predicted = np.array(predicted)

    return observed, predicted, solutions

for drug in drugs:

    print("\n")
    print("=" * 70)
    print(drug.upper())
    print("=" * 70)

    data = df[
        df["drug"] == drug
    ].copy()

    p1 = fit_model1(data)

    p2 = fit_model2(data)

    p3 = fit_model3(data)

    obs1, pred1, sol1 = evaluate_model(
        data, 1, p1
    )

    obs2, pred2, sol2 = evaluate_model(
        data, 2, p2
    )

    obs3, pred3, sol3 = evaluate_model(
        data, 3, p3
    )

    print("\nMODEL COMPARISON")

    compare_models(
        "Model 1",
        obs1,
        pred1,
        len(p1)
    )

    compare_models(
        "Model 2",
        obs2,
        pred2,
        len(p2)
    )

    compare_models(
        "Model 3",
        obs3,
        pred3,
        len(p3)
    )

    print("\nPARAMETERS")

    print("\nModel 1")
    print(f"r_S      = {p1[0]:.6f}")
    print(f"r_R      = {p1[1]:.6f}")
    print(f"alpha_S  = {p1[2]:.6f}")
    print(f"alpha_R  = {p1[3]:.6f}")

    print("\nModel 2")
    print(f"r_S      = {p2[0]:.6f}")
    print(f"r_R      = {p2[1]:.6f}")
    print(f"Emax_S   = {p2[2]:.6f}")
    print(f"EC50_S   = {p2[3]:.6f}")

    print("\nModel 3")
    print(f"r_S      = {p3[0]:.6f}")
    print(f"r_R      = {p3[1]:.6f}")
    print(f"Emax_S   = {p3[2]:.6f}")
    print(f"EC50_S   = {p3[3]:.6f}")
    print(f"k_d      = {p3[4]:.6f}")


    print("\nTIME TO RESISTANCE")

    for concentration, sol in sol1:

        t_res = tres(
            sol.t,
            sol.y[0],
            sol.y[1]
        )

        print(
            f"Model 1 | concentration={concentration:g} | "
            f"{t_res:.2f} hr"
        )

    for concentration, sol in sol2:

        t_res = tres(
            sol.t,
            sol.y[0],
            sol.y[1]
        )

        print(
            f"Model 2 | concentration={concentration:g} | "
            f"{t_res:.2f} hr"
        )

    for concentration, sol in sol3:

        t_res = tres(
            sol.t,
            sol.y[0],
            sol.y[1]
        )

        print(
            f"Model 3 | concentration={concentration:g} | "
            f"{t_res:.2f} hr"
        )

    plt.figure(figsize=(9, 6))

    for concentration in data["concentration"].unique():

        sub = data[
            data["concentration"] == concentration
        ].sort_values("time_min")

        plt.scatter(
            sub["time_min"] / 60.0,
            sub["cfu_ml"],
            s=20,
            alpha=0.35
        )


    for concentration, sol in sol1:

        plt.plot(
            sol.t,
            sol.y[0] + sol.y[1],
            "--",
            linewidth=1
        )


    for concentration, sol in sol2:

        plt.plot(
            sol.t,
            sol.y[0] + sol.y[1],
            linewidth=1
        )


    for concentration, sol in sol3:

        plt.plot(
            sol.t,
            sol.y[0] + sol.y[1],
            "-.",
            linewidth=1
        )


    plt.yscale("log")

    plt.xlabel("Time (hours)")
    plt.ylabel("Population (CFU/mL)")

    plt.title(
        f"Model Comparison - {drug.capitalize()}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"model_comparison_{drug}.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(9, 6))


    for concentration, sol in sol1:

        ratio = sol.y[1] / (
            sol.y[0] + sol.y[1] + 1e-12
        )

        plt.plot(
            sol.t,
            ratio,
            "--",
            linewidth=1
        )


    for concentration, sol in sol2:

        ratio = sol.y[1] / (
            sol.y[0] + sol.y[1] + 1e-12
        )

        plt.plot(
            sol.t,
            ratio,
            linewidth=1
        )


    for concentration, sol in sol3:

        ratio = sol.y[1] / (
            sol.y[0] + sol.y[1] + 1e-12
        )

        plt.plot(
            sol.t,
            ratio,
            "-.",
            linewidth=1
        )


    plt.axhline(
        0.5,
        linestyle=":"
    )

    plt.xlabel("Time (hours)")
    plt.ylabel("R / (S + R)")

    plt.title(
        f"Resistance Ratio - {drug.capitalize()}"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"resistance_ratio_{drug}.png",
        dpi=300
    )

    plt.show()


print("\nFinish")