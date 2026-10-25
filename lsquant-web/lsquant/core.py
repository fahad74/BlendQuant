"""LSQUANT-Web: least-squares quantitative analysis of polymer blends from IR spectra.

Python re-implementation of LSQUANT.BAS (June 19, 1994, Prof. Kap Jin Kim, Kyung Hee Univ.),
which follows Antoon/Koenig (Appl. Spectrosc. 31, 518, 1977) and Haaland/Easterling
(Appl. Spectrosc. 36, 665, 1982).  Mixture(v) = sum_j c_j * Reference_j(v).

NOTE: the original BASIC source was not available.  Method 1 (ordinary LS) is exact.
The weight functions of methods 2 and 3 are reconstructed from the papers -- edit WEIGHTS
below if you need to match the DOS program's output exactly.
"""
import re
import numpy as np

# weight per data point as a function of mixture absorbance A (normalised to mean 1 later)
WEIGHTS = {
    "ols":     lambda A: np.ones_like(A),
    "haaland": lambda A: 10.0 ** (-2.0 * np.clip(A, 0.0, None)),   # var(A) ~ 10^(2A) for constant T-noise
    "antoon":  lambda A: 1.0 / np.clip(A, 1e-3, None) ** 2,        # equal relative error per point
}
METHODS = {
    "ols": "1. Non-weighted least squares (Haaland)",
    "haaland": "2. Weighted least squares (Haaland)",
    "antoon": "3. Weighted least squares (Antoon)",
}


def read_spectrum(raw, name=""):
    """Read .ASP (N / first / last / 3 flags / N values) or two-column .PRN/.TXT/.CSV."""
    text = raw.decode("latin1") if isinstance(raw, (bytes, bytearray)) else raw
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    try:
        n = int(float(lines[0]))
        if len(lines) >= 6 + n and all(len(l.split()) == 1 for l in lines[: 6 + n]):
            x = np.linspace(float(lines[1]), float(lines[2]), n)
            return x, np.array(lines[6 : 6 + n], float)
    except (ValueError, IndexError):
        pass
    pts = []
    for l in lines:
        t = re.split(r"[\s,;]+", l)
        try:
            pts.append((float(t[0]), float(t[1])))
        except (ValueError, IndexError):
            continue
    if len(pts) < 10:
        raise ValueError(f"{name or 'file'}: not a recognised spectrum file")
    x, y = np.array(pts).T
    return x, y


def _on_grid(x, y, xg):
    o = np.argsort(x)
    return np.interp(xg, x[o], y[o], left=np.nan, right=np.nan)


def analyse(xm, ym, refs, names, lo, hi, amin=-np.inf, amax=np.inf, clo=None, chi=None):
    """refs: list of (x, y).  Returns dict method -> results."""
    o = np.argsort(xm); xm, ym = xm[o], ym[o]
    R = np.array([_on_grid(x, y, xm) for x, y in refs])           # references on mixture grid
    ok = ~np.isnan(R).any(axis=0)
    sel = ok & (xm >= lo) & (xm <= hi) & (ym >= amin) & (ym <= amax)
    n, p = int(sel.sum()), len(refs)
    if n <= p:
        raise ValueError("Too few data points inside the fit range / absorbance thresholds.")
    cmask = ok & (xm >= (clo if clo is not None else lo)) & (xm <= (chi if chi is not None else hi))
    out = {}
    for key, wf in WEIGHTS.items():
        w = wf(ym[sel]); w = w / w.mean(); sw = np.sqrt(w)
        X = R[:, sel].T * sw[:, None]
        c, *_ = np.linalg.lstsq(X, ym[sel] * sw, rcond=None)
        res = ym[sel] * sw - X @ c
        se = float(np.sqrt(res @ res / (n - p)))
        cse = se * np.sqrt(np.diag(np.linalg.pinv(X.T @ X)))
        fit = np.where(ok, R.T @ c, np.nan)
        r = float(np.corrcoef(fit[cmask], ym[cmask])[0, 1]) if cmask.sum() > 2 else float("nan")
        out[key] = dict(content=dict(zip(names, c)), content_se=dict(zip(names, cse)),
                        weight_pct=dict(zip(names, 100 * c / c.sum())), std_error=se,
                        corr=r, n_points=n, fit=fit, x=xm, y=ym, used=sel)
    return out
