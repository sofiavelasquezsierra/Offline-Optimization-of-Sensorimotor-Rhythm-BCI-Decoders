import numpy as np 
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def _concat_cov(epochs, demean=True):
    """
    estimate class covariance on concatenated epochs.
    epochs: list of arrays, each (n_chans, n_samples)
    Returns: (n_chans, n_chans) covariance.
    """
    # Concatenate along time
    X = np.concatenate([
        (e - e.mean(axis=-1, keepdims=True)) if demean else e
        for e in epochs
    ], axis=1)  # (n_ch, total_T)
    # Sample covariance 
    C = (X @ X.T) / (X.shape[1] - 1)
    # Symmetrize (numerical hygiene)
    C = 0.5 * (C + C.T)
    return C


def csp_fit(class1_epochs, class2_epochs, reg=1e-6, m_pairs=3):
    """
    Fit CSP filters:
      - covariance per class from concatenated epochs
      - generalized eigendecomposition: C1 v = λ (C1 + C2) v
      - two-class component order by descending |λ - 0.5|
    Returns:
      W: filters, shape (n_ch, 2*m_pairs)
    """
    C1 = _concat_cov(class1_epochs)
    C2 = _concat_cov(class2_epochs)
    Csum = C1 + C2

    n_ch = C1.shape[0]
    # Light ridge to ensure positive-definiteness
    scale1 = np.trace(C1) / n_ch
    scales = np.trace(Csum) / n_ch
    C1r  = C1  + reg * scale1 * np.eye(n_ch)
    Csumr = Csum + reg * scales * np.eye(n_ch)

    # Solve generalized eigenproblem C1 v = λ (C1 + C2) v
    # we want to pass C1r and Csumr
    lam, V = eigh(C1r, Csumr)   

    # goal of CSP: find filters that maximize the variance in one class while minimizing it in the other
    # the most discriminative filters have λ far from 0.5
    order = np.argsort(np.abs(lam - 0.5))[::-1] # order by decreasing |λ - 0.5|
    V = V[:, order]

    # Take top 2*m_pairs components
    # We want to select the top filters (2*m_pairs)
    W = V[:, :2*m_pairs]   # filters
    return W


def csp_transform(W, epochs, log=True):
    """
    Project epochs with CSP filters and return average band power with optional log-transform.
    epochs: list of (n_ch, n_samp)
    Returns: (n_trials, n_components)
    """
    feats = []
    for e in epochs:
        # Use matrix multiplication W^T and epoch data
        Z = W.T @ e         # Project epochs with CSP filters (n_comp, T)
        p = (Z ** 2).mean(axis=1)  # average power
        if log: # log-transform (optional)
            p = np.log(p + 1e-12) 
        feats.append(p)
    return np.asarray(feats)


def plot_csp_feature_extraction(
    epochs, y,
    W,                 # CSP filters 
    log_power=True
):
    """
    Visualize CSP feature extraction on 2-class data.

    Parameters
    ----------
    epochs : list[np.ndarray]
        Each trial array is (n_ch, n_time), already preprocessed & windowed.
    y : array-like of shape (n_trials,)
        Binary labels (e.g., {1,2}).
    log_power : bool
        Apply log to average power (both panels) for consistency.

    Returns
    -------
    fig : matplotlib Figure
    (X_raw, X_csp) : tuple of np.ndarray
        2D raw powers and 2D CSP powers used for plotting.
    """
    classes = np.unique(y)

    ch1, ch2 = (25, 29) # raw channels to show before CSP; here we choose C3 and C4 for visualization

    # -------- Raw 2D features (mean power in two chosen channels) --------
    X_raw = []
    for e in epochs:
        p1 = (e[ch1] ** 2).mean()
        p2 = (e[ch2] ** 2).mean()
        if log_power:
            p1 = np.log(p1 + 1e-12)
            p2 = np.log(p2 + 1e-12)
        X_raw.append([p1, p2])
    X_raw = np.asarray(X_raw)

    # -------- CSP 2D features (first two CSP components) --------
    feats = csp_transform(W, epochs, log=log_power)  # (n_trials, n_comp)
    if feats.shape[1] < 2:
        # pad to 2D if user only asked for 1 component
        feats = np.pad(feats, ((0, 0), (0, 2 - feats.shape[1])), constant_values=0.0)
    X_csp = feats[:, :2]

    # -------- Plotting --------
    title_before="Before CSP filtering",
    title_after="After CSP filtering",
    figsize=(6, 9)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    # Before CSP
    ax = axes[0]
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(X_raw[m, 0], X_raw[m, 1],
                   s=18, alpha=0.75, marker='x' if i == 0 else 'o',
                   label=f"class {c}")
    ax.set_title(title_before)
    ax.set_xlabel(f"Channel {ch1+1} power")
    ax.set_ylabel(f"Channel {ch2+1} power")
    ax.legend(loc="best")

    # After CSP
    ax = axes[1]
    for i, c in enumerate(classes):
        m = (y == c)
        ax.scatter(X_csp[m, 0], X_csp[m, 1],
                   s=18, alpha=0.75, marker='x' if i == 0 else 'o',
                   label=f"class {c}")
    ax.set_title(title_after)
    ax.set_xlabel("CSP component 1 power")
    ax.set_ylabel("CSP component 2 power")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig, (X_raw, X_csp)
