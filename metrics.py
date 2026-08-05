import numpy as np

def rss(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = rss(y_true, y_pred)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot

def aic(y_true, y_pred, k):
    n = len(y_true)
    sse = max(rss(y_true, y_pred), 1e-12)
    return n * np.log(sse / n) + 2 * k

def tres(time, S, R, threshold=0.5):
    ratio = R / (S + R + 1e-12)
    idx = np.where(ratio >= threshold)[0]

    if len(idx) == 0:
        return np.inf
    return time[idx[0]]

def compare_models(name, y_true, y_pred, k):
    print(f"\n{name}")
    print(f"R²   : {r2(y_true, y_pred):.4f}")
    print(f"RMSE : {rmse(y_true, y_pred):.4f}")
    print(f"AIC  : {aic(y_true, y_pred, k):.2f}")