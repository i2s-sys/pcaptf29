import numpy as np
from scipy.stats import spearmanr


def inf_fs_u(X, alpha=0.5, verbose=0):
    """
    Inf-FS Unsupervised Feature Selection (Python version)

    Parameters
    ----------
    X : ndarray, shape (T, n)
        Input data matrix (T samples x n features).
    alpha : float, optional
        Mixing coefficient in range [0,1]. Default=0.5.
    verbose : int, optional
        Verbosity flag (0 or 1).

    Returns
    -------
    RANKED : ndarray
        Indices of features ordered by importance (descending).
    WEIGHT : ndarray
        Feature weights (importance scores).
    SUBSET : ndarray
        Selected subset of top features.
    """

    print("\n+ Feature selection method: Inf-FS Unsupervised (2020)")

    # 1) Priors/weights estimation
    if verbose:
        print("1) Priors/weights estimation")

    # Spearman correlation matrix
    corr_ij, _ = spearmanr(X)
    if corr_ij.ndim == 2 and corr_ij.shape[0] != X.shape[1]:
        # spearmanr returns full (samples+features) correlation if axis not specified
        corr_ij = corr_ij[-X.shape[1]:, -X.shape[1]:]
    corr_ij = 1 - np.abs(corr_ij)
    corr_ij = np.nan_to_num(corr_ij, nan=0.0, posinf=0.0, neginf=0.0)

    # Standard deviation
    STD = np.std(X, axis=0)
    STDMatrix = np.maximum.outer(STD, STD)
    STDMatrix = STDMatrix - np.min(STDMatrix)
    if np.max(STDMatrix) > 0:
        sigma_ij = STDMatrix / np.max(STDMatrix)
    else:
        sigma_ij = STDMatrix
    sigma_ij = np.nan_to_num(sigma_ij, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Building graph
    if verbose:
        print("2) Building the graph G = <V,E>")

    N = X.shape[1]
    eps = 5e-06 * N
    factor = 1 - eps

    A = alpha * sigma_ij + (1 - alpha) * corr_ij
    rho = np.max(np.sum(A, axis=1))

    # Substochastic rescaling
    A = A / (np.max(np.sum(A, axis=1)) + eps)
    assert np.max(np.sum(A, axis=1)) < 1.0

    # 3) Inf-FS Core
    I = np.eye(A.shape[0])
    r = factor / rho
    y = I - (r * A)
    S = np.linalg.inv(y)

    # 4) Energy scores
    WEIGHT = np.sum(S, axis=1)

    # 5) Ranking
    RANKED = np.argsort(-WEIGHT)
    WEIGHT = WEIGHT

    # 6) Subset selection
    e = np.ones(N)
    t = S @ e

    nbins = int(0.5 * N)
    counts, _ = np.histogram(t, bins=nbins)
    thr = np.mean(counts)
    size_sub = np.sum(counts > thr)

    print(f"Inf-FS (U) Nb. Features Selected = {size_sub:.4f} ({100 * size_sub / N:.2f}%)")

    SUBSET = RANKED[:size_sub]

    return RANKED, WEIGHT, SUBSET
