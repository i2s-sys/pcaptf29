import numpy as np
from sklearn.feature_selection import mutual_info_classif

def inf_fs_s(X, Y, alpha):
    """
    Inf-FS Supervised Feature Selection (Python version)

    Parameters
    ----------
    X : ndarray, shape (T, n)
        Input data matrix (T samples x n features).
    Y : ndarray, shape (T,)
        Labels (e.g., -1, 1).
    alpha : list or tuple of 3 floats
        Mixing coefficients [alpha1, alpha2, alpha3].

    Returns
    -------
    RANKED : ndarray
        Indices of features ordered by importance (descending).
    WEIGHT : ndarray
        Feature weights (importance scores).
    SUBSET : ndarray
        Selected subset of top features.
    """

    print("\n+ Feature selection method: Inf-FS Supervised (2020)")
    N = X.shape[1]

    eps = 5e-06 * N
    factor = 1 - eps  # shrinking

    # Build graph
    A, rho = get_graph_weights(X, Y, alpha, eps)

    # Inf-FS core
    I = np.eye(A.shape[0])
    r = factor / rho
    y = I - (r * A)
    S = np.linalg.inv(y)

    # Energy scores
    WEIGHT = np.sum(S, axis=1)

    # Ranking
    RANKED = np.argsort(-WEIGHT)  # descending
    WEIGHT = WEIGHT

    # Subset selection
    e = np.ones(N)
    t = S @ e

    nbins = int(0.5 * N)
    counts, _ = np.histogram(t, bins=nbins)
    thr = np.mean(counts)
    size_sub = np.sum(counts > thr)

    print(f"Inf-FS (S) Nb. Features Selected = {size_sub:.4f} ({100 * size_sub / N:.2f}%)")

    SUBSET = RANKED[:size_sub]

    return RANKED, WEIGHT, SUBSET


def get_graph_weights(train_x, train_y, alpha, eps):
    """
    Build the supervised graph (Python version)
    """

    # Metric 1: Mutual Information
    mi_s = mutual_info_classif(train_x, train_y, discrete_features=False)
    mi_s = np.nan_to_num(mi_s, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero-Max norm
    if np.max(mi_s) > 0:
        mi_s = (mi_s - np.min(mi_s)) / (np.max(mi_s) - np.min(mi_s))

    # Metric 2: Class separation
    mean_pos = np.mean(train_x[train_y == 1, :], axis=0)
    mean_neg = np.mean(train_x[train_y == -1, :], axis=0)
    fi_s = (mean_pos - mean_neg) ** 2

    st = np.var(train_x[train_y == 1, :], axis=0) + np.var(train_x[train_y == -1, :], axis=0)
    st[st == 0] = 10000
    fi_s = fi_s / st
    fi_s = np.nan_to_num(fi_s, nan=0.0, posinf=0.0, neginf=0.0)

    if np.max(fi_s) > 0:
        fi_s = (fi_s - np.min(fi_s)) / (np.max(fi_s) - np.min(fi_s))

    # Standard deviation
    std_s = np.std(train_x, axis=0)
    std_s = np.nan_to_num(std_s, nan=0.0, posinf=0.0, neginf=0.0)

    SD = np.maximum.outer(std_s, std_s)
    if np.max(SD) > 0:
        SD = (SD - np.min(SD)) / (np.max(SD) - np.min(SD))

    # Build adjacency matrix
    MI = np.tile(mi_s, (len(mi_s), 1))
    FI = np.tile(fi_s, (len(fi_s), 1))

    G = (
        alpha[0] * ((MI + FI.T) / 2)
        + alpha[1] * ((MI + SD.T) / 2)
        + alpha[2] * ((SD + FI.T) / 2)
    )

    rho = np.max(np.sum(G, axis=1))

    # Substochastic rescaling
    G = G / (np.max(np.sum(G, axis=1)) + eps)
    assert np.max(np.sum(G, axis=1)) < 1.0

    return G, rho
