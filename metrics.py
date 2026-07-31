import numpy as np

def calculate_rss(y_true, y_pred):
    """Residual Sum of Squares"""
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)

def calculate_r2(y_true, y_pred):
    """Coefficient of Determination (R^2)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rss = calculate_rss(y_true, y_pred)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    if tss == 0:
        return 0.0
    return 1.0 - (rss / tss)

def calculate_aic(y_true, y_pred, k):
    """Akaike Information Criterion (AIC)

    k: จำนวนพารามิเตอร์ที่ถูกประมาณค่า (Fitted Parameters)
                """
    y_true = np.array(y_true)
    n = len(y_true)
    rss = calculate_rss(y_true, y_pred)

    rss_safe = max(rss, 1e-12)
    return 2 * k + n * np.log(rss_safe / n)

def calculate_tres(t_eval, S_vec, R_vec, threshold=0.5):
    """Tres
    จุดที่สัดส่วน R / (S + R) >= threshold (default = 0.5)
    """
    total = S_vec + R_vec
                        
    total_safe = np.where(total == 0, 1e-12, total)
    ratio = R_vec / total_safe

    res_indices = np.where(ratio >= threshold)[0]
    if len(res_indices) > 0:
        return t_eval[res_indices[0]]
    else:
        return np.inf 