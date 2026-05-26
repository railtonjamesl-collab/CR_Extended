# -*- coding: utf-8 -*-
"""
Created on Fri May 22 07:57:48 2026

@author: railt
"""

import time
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from functions import generate_circle_100grid, compute_boundary, generate_gradient, rademacher
import matplotlib.pyplot as plt
import numpy as np

SIGNAL_TYPES = ["circle"]
SIGNAL_FWHMS = [3]
NOISE_FWHMS = [3]
ALPHAS = [0.1]
SAMPLES = [25,50,100,250,500]
BOOT_REP = 500
THR = 2
REPEAT = 500
NOISE_SCALE = [1.0]

RADII = [30]
MAGNITUDES = [3]

MAXSIGS = [3]
MINSIG = 0.0


def fwhm_to_sigma(fwhm):
    if fwhm <= 0:
        return 0.0
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def contour_points(field, thr):
    mask = field > thr
    _, s1, s0, *_ = compute_boundary(mask)

    s0 = np.asarray(s0, dtype=int)
    s1 = np.asarray(s1, dtype=int)

    if s0.size == 0 or s1.size == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=int),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    b0 = field[s0[:, 0], s0[:, 1]]
    b1 = field[s1[:, 0], s1[:, 1]]
    diff = b1 - b0


    lamda = (thr - b0) / diff #Compute M2
    points = (1.0 - lamda)[:, None] * s0 + lamda[:, None] * s1
    return points, s0, s1, lamda


def directed_hausdorff_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    return directed_hausdorff(a, b)[0]


def symmetric_hausdorff(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    return max(
        directed_hausdorff(a, b)[0],
        directed_hausdorff(b, a)[0],
    )


def make_signal(signal_type, signal_fwhm, radius=None, magnitude=None, maxsig=None):
    if signal_type == "circle":
        signal = generate_circle_100grid(radius, magnitude)[0]
    elif signal_type == "grad":
        signal = generate_gradient(MINSIG, maxsig)
    else:
        raise ValueError("signal_type must be 'circle' or 'grad'")

    sigma_signal = fwhm_to_sigma(signal_fwhm) #generate a sigma
    if sigma_signal > 0:
        signal = ndimage.gaussian_filter(signal, sigma_signal) # only apply filter when we have positive smoothing

    return signal


def simulate_instances(signal, n_subject, noise_fwhm, noise_scale):
    sigma_noise = fwhm_to_sigma(noise_fwhm)

    epsilon = np.random.randn(n_subject, 100, 100)

    if sigma_noise > 0:
        epsilon = ndimage.gaussian_filter(
            epsilon,
            sigma=(0, sigma_noise, sigma_noise)
        )

    epsilon /= np.std(epsilon, axis=(1, 2), ddof=1, keepdims=True)

    instances = signal[None, :, :] + noise_scale * epsilon

    return instances


def bootstrap_directed_radius_rademacher(instances, thr, boot_rep):
    n_subject = instances.shape[0] # number of subject

    coefficient = np.mean(instances, axis=0) #mu hat
    est_bdry, _, _, _ = contour_points(coefficient, thr) #estimated bdry interpolated

    if len(est_bdry) == 0: #no threshold so return early
        return None, est_bdry, coefficient

    residual = instances - coefficient #residual
    boot_stats = []

    for _ in range(boot_rep):
        ri = rademacher(n_subject)
        boot_residual = np.sum(ri[:, None, None] * residual, axis=0) / n_subject
        coefficient_star = coefficient + boot_residual

        bdry_star, _, _, _ = contour_points(coefficient_star, thr)
        if len(bdry_star) == 0:
            continue

        stat = directed_hausdorff_distance(est_bdry, bdry_star)
        """
        d2 = directed_hausdorff_distance(bdry_star, est_bdry)
        stat = max(stat, d2)
        """
        boot_stats.append(stat)

    boot_stats = np.asarray(boot_stats, dtype=float)
    return boot_stats, est_bdry, coefficient


def run_one_rep(thr, n_subject, boot_rep, signal_type, signal_fwhm,
                noise_fwhm, noise_scale, radius=None, magnitude=None, maxsig=None):
    #make signal
    signal = make_signal(
        signal_type=signal_type,
        signal_fwhm=signal_fwhm,
        radius=radius,
        magnitude=magnitude,
        maxsig=maxsig,
    )
    #make samples
    instances = simulate_instances(
        signal=signal,
        n_subject=n_subject,
        noise_fwhm=noise_fwhm,
        noise_scale=noise_scale,
    )
    #compute boundary, and check if length 0 or not
    true_bdry, _, _, _ = contour_points(signal, thr)
    """
    if len(true_bdry) == 0:
        return {
            "valid": False,
            "reason": "true contour extraction failed or no boundary",
        }
    """
    #bootstrapping
    boot_stats, est_bdry, coefficient = bootstrap_directed_radius_rademacher(
        instances=instances,
        thr=thr,
        boot_rep=boot_rep,
    )
    """
    if boot_stats is None or len(est_bdry) == 0:
        return {
            "valid": False,
            "reason": "estimated contour extraction failed",
        }

    if len(boot_stats) == 0:
        return {
            "valid": False,
            "reason": "all bootstrap contours failed",
        }
    """
    return {
        "valid": True,
        "boot_stats": boot_stats,
        "est_bdry": est_bdry,
        "true_bdry": true_bdry,
    }

def numerical_test(repeat, alpha, signal_type, threshold, signal_fwhm,
                   noise_fwhm, samples, boot_rep, noise_scale=1.0,
                   radius=None, magnitude=None, maxsig=None):

    cover_directed = []
    cover_symmetric = []
    d_true_to_est_all = []
    d_est_to_true_all = []
    rel_asym_all = []
    runtimes = []

    for _ in range(repeat):
        tic = time.perf_counter()

        result = run_one_rep(
            thr=threshold,
            n_subject=samples,
            boot_rep=boot_rep,
            signal_type=signal_type,
            signal_fwhm=signal_fwhm,
            noise_fwhm=noise_fwhm,
            noise_scale=noise_scale,
            radius=radius,
            magnitude=magnitude,
            maxsig=maxsig,
        )

        if not result["valid"]:
            continue

        boot_stats = result["boot_stats"]
        est_bdry = result["est_bdry"]
        true_bdry = result["true_bdry"]

        radius_hat = np.quantile(boot_stats, 1 - alpha)

        d_true_to_est = directed_hausdorff_distance(true_bdry, est_bdry)
        d_est_to_true = directed_hausdorff_distance(est_bdry, true_bdry)
        h_sym = max(d_true_to_est, d_est_to_true)

        rel_asym = abs(d_true_to_est - d_est_to_true) / h_sym

        cover_directed.append(int(d_true_to_est <= radius_hat))
        cover_symmetric.append(int(h_sym <= radius_hat))
        d_true_to_est_all.append(float(d_true_to_est))
        d_est_to_true_all.append(float(d_est_to_true))
        rel_asym_all.append(float(rel_asym))
        runtimes.append(time.perf_counter() - tic)

    n_valid = len(cover_directed)

    if n_valid == 0:
        return {
            "n_valid": 0,
            "coverage_directed": np.nan,
            "coverage_symmetric": np.nan,
            "d_true_to_est_mean": np.nan,
            "d_true_to_est_sd": np.nan,
            "d_est_to_true_mean": np.nan,
            "d_est_to_true_sd": np.nan,
            "rel_asym_mean": np.nan,
            "rel_asym_sd": np.nan,
            "runtime_mean": np.nan,
            "runtime_sd": np.nan,
        }

    return {
        "n_valid": n_valid,
        "coverage_directed": np.mean(cover_directed),
        "coverage_symmetric": np.mean(cover_symmetric),
        "d_true_to_est_mean": np.mean(d_true_to_est_all),
        "d_true_to_est_sd": np.std(d_true_to_est_all, ddof=1) if n_valid > 1 else 0.0,
        "d_est_to_true_mean": np.mean(d_est_to_true_all),
        "d_est_to_true_sd": np.std(d_est_to_true_all, ddof=1) if n_valid > 1 else 0.0,
        "rel_asym_mean": np.mean(rel_asym_all),
        "rel_asym_sd": np.std(rel_asym_all, ddof=1) if n_valid > 1 else 0.0,
        "runtime_mean": np.mean(runtimes),
        "runtime_sd": np.std(runtimes, ddof=1) if n_valid > 1 else 0.0,
    }


"""
MAIN RUN START HERE
"""
results = []

for signal in SIGNAL_TYPES:
    for signal_fwhm in SIGNAL_FWHMS:
        for noise_fwhm in NOISE_FWHMS:
            for alpha in ALPHAS:
                for n in SAMPLES:
                    for noise_scale in NOISE_SCALE:
                        if signal == "circle":
                            for radius in RADII:
                                for magnitude in MAGNITUDES:
                                    if THR >= magnitude:
                                        continue
    
                                    print(
                                        f"Running: signal={signal}, signal_fwhm={signal_fwhm}, "
                                        f"noise_fwhm={noise_fwhm}, alpha={alpha}, samples={n}, "
                                        f"radius={radius}, magnitude={magnitude}, noise scale = {noise_scale}"
                                    )
    
                                    out = numerical_test(
                                        repeat=REPEAT,
                                        alpha=alpha,
                                        signal_type=signal,
                                        threshold=THR,
                                        signal_fwhm=signal_fwhm,
                                        noise_fwhm=noise_fwhm,
                                        samples=n,
                                        boot_rep=BOOT_REP,
                                        noise_scale=noise_scale,
                                        radius=radius,
                                        magnitude=magnitude,
                                    )
    
                                    row = {
                                        "signal_type": signal,
                                        "signal_fwhm": signal_fwhm,
                                        "noise_fwhm": noise_fwhm,
                                        "alpha": alpha,
                                        "sample": n,
                                        "threshold": THR,
                                        "boot_rep": BOOT_REP,
                                        "repeat": REPEAT,
                                        "noise_scale": noise_scale,
                                        "radius_param": radius,
                                        "magnitude": magnitude,
                                        "maxsig": np.nan,
                                        "n_valid": out["n_valid"],
                                        "coverage_directed": out["coverage_directed"],
                                        "coverage_symmetric": out["coverage_symmetric"],
                                        "d_true_to_est_mean": out["d_true_to_est_mean"],
                                        "d_true_to_est_sd": out["d_true_to_est_sd"],
                                        "d_est_to_true_mean": out["d_est_to_true_mean"],
                                        "d_est_to_true_sd": out["d_est_to_true_sd"],
                                        "rel_asym_mean": out["rel_asym_mean"],
                                        "rel_asym_sd": out["rel_asym_sd"],
                                        "runtime_mean": out["runtime_mean"],
                                        "runtime_sd": out["runtime_sd"],
                                    }
                                    results.append(row)
                                    print(out["coverage_directed"], out["coverage_symmetric"])
    
                        elif signal == "grad":
                            for maxsig in MAXSIGS:
                                threshold = 5 if maxsig == 10 else THR

                                if threshold >= maxsig:
                                    continue
                                if THR >= maxsig:
                                    continue
    
                                print(
                                    f"Running: signal={signal}, signal_fwhm={signal_fwhm}, "
                                    f"noise_fwhm={noise_fwhm}, alpha={alpha}, samples={n}, "
                                    f"maxsig={maxsig}, threshold={threshold}"
                                )
                                
                                out = numerical_test(
                                    repeat=REPEAT,
                                    alpha=alpha,
                                    signal_type=signal,
                                    threshold=threshold,
                                    signal_fwhm=signal_fwhm,
                                    noise_fwhm=noise_fwhm,
                                    samples=n,
                                    boot_rep=BOOT_REP,
                                    noise_scale=noise_scale,
                                    maxsig=maxsig,
                                )
    
                                row = {
                                    "signal_type": signal,
                                    "signal_fwhm": signal_fwhm,
                                    "noise_fwhm": noise_fwhm,
                                    "alpha": alpha,
                                    "sample": n,
                                    "threshold": threshold,
                                    "boot_rep": BOOT_REP,
                                    "repeat": REPEAT,
                                    "noise_scale": noise_scale,
                                    "radius_param": np.nan,
                                    "magnitude": np.nan,
                                    "maxsig": maxsig,
                                    "n_valid": out["n_valid"],
                                    "coverage_directed": out["coverage_directed"],
                                    "coverage_symmetric": out["coverage_symmetric"],
                                    "d_true_to_est_mean": out["d_true_to_est_mean"],
                                    "d_true_to_est_sd": out["d_true_to_est_sd"],
                                    "d_est_to_true_mean": out["d_est_to_true_mean"],
                                    "d_est_to_true_sd": out["d_est_to_true_sd"],
                                    "rel_asym_mean": out["rel_asym_mean"],
                                    "rel_asym_sd": out["rel_asym_sd"],
                                    "runtime_mean": out["runtime_mean"],
                                    "runtime_sd": out["runtime_sd"],
                                }
                                results.append(row)
                                print(out["coverage_directed"], out["coverage_symmetric"])


df = pd.DataFrame(results)

plt.plot(
    df["sample"],
    df["coverage_directed"],
    label="Directed coverage"
)
"""
os.makedirs("results", exist_ok=True)
out_csv = "results/numerical_hausdorff_directasd.csv"
df.to_csv(out_csv, index=False)
print(f"Saved results to {out_csv}")
"""
"""
Asymptotic Hausdorff experiment
"""
"""
import time
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from functions import generate_circle_100grid, compute_boundary, generate_gradient


SIGNAL_TYPES = ["circle", "grad"]
SIGNAL_FWHMS = [3]
NOISE_FWHMS = [3]
ALPHAS = [0.1]
SAMPLES = [100, 400, 1000, 5000]
BOOT_REP = 500
REPEAT = 1000
NOISE_SCALE = [1]

# gradient settings
MAXSIGS = [3, 100]
THRESHOLD_FRACS = [0.5]
MINSIG = 0.0


def fwhm_to_sigma(fwhm):
    if fwhm <= 0:
        return 0.0
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def rademacher(n):
    return np.random.choice([-1, 1], size=n)


def contour_points(field, thr):
    mask = field > thr
    _, s1, s0, *_ = compute_boundary(mask)

    s0 = np.asarray(s0, dtype=int)
    s1 = np.asarray(s1, dtype=int)

    if s0.size == 0 or s1.size == 0:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=int),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    b0 = field[s0[:, 0], s0[:, 1]]
    b1 = field[s1[:, 0], s1[:, 1]]
    diff = b1 - b0

    keep = np.abs(diff) > 1e-12
    if not np.any(keep):
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=int),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    s0 = s0[keep]
    s1 = s1[keep]
    b0 = b0[keep]
    diff = diff[keep]

    lam = (thr - b0) / diff
    pts = (1.0 - lam)[:, None] * s0 + lam[:, None] * s1
    return pts, s0, s1, lam


def directed_hausdorff_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    return directed_hausdorff(a, b)[0]


def symmetric_hausdorff(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.inf
    return max(
        directed_hausdorff(a, b)[0],
        directed_hausdorff(b, a)[0],
    )


def make_signal(signal_type, signal_fwhm, radius=None, magnitude=None, maxsig=None):
    if signal_type == "circle":
        signal = generate_circle_100grid(radius, magnitude)[0]
    elif signal_type == "grad":
        signal = generate_gradient(MINSIG, maxsig)
    else:
        raise ValueError("signal_type must be 'circle' or 'grad'")

    sigma_signal = fwhm_to_sigma(signal_fwhm)
    if sigma_signal > 0:
        signal = ndimage.gaussian_filter(signal, sigma_signal)

    return signal


def simulate_instances(signal, n_subject, noise_fwhm, noise_scale):
    h, w = signal.shape
    instances = np.empty((n_subject, h, w), dtype=float)

    sigma_noise = fwhm_to_sigma(noise_fwhm)

    for i in range(n_subject):
        epsilon = np.random.randn(h, w)
        if sigma_noise > 0:
            epsilon = ndimage.gaussian_filter(epsilon, sigma_noise)

        epsilon /= max(np.std(epsilon, ddof=1), 1e-12)
        instances[i] = signal + noise_scale * epsilon

    return instances


def bootstrap_directed_radius_rademacher(instances, thr, boot_rep):
    n_subject = instances.shape[0]

    coefficient = np.mean(instances, axis=0)
    est_bdry, _, _, _ = contour_points(coefficient, thr)

    if len(est_bdry) == 0:
        return None, est_bdry, coefficient

    centered = instances - coefficient
    boot_stats = []

    for _ in range(boot_rep):
        xi = rademacher(n_subject)
        fluctuation = np.tensordot(xi, centered, axes=(0, 0)) / n_subject
        coefficient_star = coefficient + fluctuation

        bdry_star, _, _, _ = contour_points(coefficient_star, thr)
        if len(bdry_star) == 0:
            continue

        stat = directed_hausdorff_distance(est_bdry, bdry_star)
        if np.isfinite(stat):
            boot_stats.append(stat)

    boot_stats = np.asarray(boot_stats, dtype=float)
    return boot_stats, est_bdry, coefficient


def run_one_rep(thr, n_subject, boot_rep, signal_type, signal_fwhm,
                noise_fwhm, noise_scale, radius=None, magnitude=None, maxsig=None):
    signal = make_signal(
        signal_type=signal_type,
        signal_fwhm=signal_fwhm,
        radius=radius,
        magnitude=magnitude,
        maxsig=maxsig,
    )

    instances = simulate_instances(
        signal=signal,
        n_subject=n_subject,
        noise_fwhm=noise_fwhm,
        noise_scale=noise_scale,
    )

    true_bdry, _, _, _ = contour_points(signal, thr)
    if len(true_bdry) == 0:
        return {"valid": False, "reason": "true contour extraction failed"}

    boot_stats, est_bdry, coefficient = bootstrap_directed_radius_rademacher(
        instances=instances,
        thr=thr,
        boot_rep=boot_rep,
    )

    if boot_stats is None or len(est_bdry) == 0:
        return {"valid": False, "reason": "estimated contour extraction failed"}

    if len(boot_stats) == 0:
        return {"valid": False, "reason": "all bootstrap contours failed"}

    return {
        "valid": True,
        "boot_stats": boot_stats,
        "est_bdry": est_bdry,
        "true_bdry": true_bdry,
    }


def numerical_test(repeat, alpha, signal_type, threshold, signal_fwhm,
                   noise_fwhm, samples, boot_rep, noise_scale=1.0,
                   radius=None, magnitude=None, maxsig=None):

    cover_directed = []
    cover_symmetric = []
    d_true_to_est_all = []
    d_est_to_true_all = []
    rel_asym_all = []
    h_sym_all = []
    runtimes = []

    for _ in range(repeat):
        tic = time.perf_counter()

        result = run_one_rep(
            thr=threshold,
            n_subject=samples,
            boot_rep=boot_rep,
            signal_type=signal_type,
            signal_fwhm=signal_fwhm,
            noise_fwhm=noise_fwhm,
            noise_scale=noise_scale,
            radius=radius,
            magnitude=magnitude,
            maxsig=maxsig,
        )

        if not result["valid"]:
            continue

        boot_stats = result["boot_stats"]
        est_bdry = result["est_bdry"]
        true_bdry = result["true_bdry"]

        radius_hat = np.quantile(boot_stats, 1 - alpha)

        d_true_to_est = directed_hausdorff_distance(true_bdry, est_bdry)
        d_est_to_true = directed_hausdorff_distance(est_bdry, true_bdry)
        h_sym = max(d_true_to_est, d_est_to_true)
        rel_asym = abs(d_true_to_est - d_est_to_true) / max(h_sym, 1e-12)

        cover_directed.append(int(d_true_to_est <= radius_hat))
        cover_symmetric.append(int(h_sym <= radius_hat))
        d_true_to_est_all.append(float(d_true_to_est))
        d_est_to_true_all.append(float(d_est_to_true))
        h_sym_all.append(float(h_sym))
        rel_asym_all.append(float(rel_asym))
        runtimes.append(time.perf_counter() - tic)

    n_valid = len(cover_directed)

    if n_valid == 0:
        return {
            "n_valid": 0,
            "coverage_directed": np.nan,
            "coverage_symmetric": np.nan,
            "d_true_to_est_mean": np.nan,
            "d_true_to_est_sd": np.nan,
            "d_est_to_true_mean": np.nan,
            "d_est_to_true_sd": np.nan,
            "h_sym_mean": np.nan,
            "h_sym_sd": np.nan,
            "rel_asym_mean": np.nan,
            "rel_asym_sd": np.nan,
            "runtime_mean": np.nan,
            "runtime_sd": np.nan,
        }

    return {
        "n_valid": n_valid,
        "coverage_directed": np.mean(cover_directed),
        "coverage_symmetric": np.mean(cover_symmetric),
        "d_true_to_est_mean": np.mean(d_true_to_est_all),
        "d_true_to_est_sd": np.std(d_true_to_est_all, ddof=1) if n_valid > 1 else 0.0,
        "d_est_to_true_mean": np.mean(d_est_to_true_all),
        "d_est_to_true_sd": np.std(d_est_to_true_all, ddof=1) if n_valid > 1 else 0.0,
        "h_sym_mean": np.mean(h_sym_all),
        "h_sym_sd": np.std(h_sym_all, ddof=1) if n_valid > 1 else 0.0,
        "rel_asym_mean": np.mean(rel_asym_all),
        "rel_asym_sd": np.std(rel_asym_all, ddof=1) if n_valid > 1 else 0.0,
        "runtime_mean": np.mean(runtimes),
        "runtime_sd": np.std(runtimes, ddof=1) if n_valid > 1 else 0.0,
    }


results = []

for signal in SIGNAL_TYPES:
    for signal_fwhm in SIGNAL_FWHMS:
        for noise_fwhm in NOISE_FWHMS:
            for alpha in ALPHAS:
                for n in SAMPLES:
                    for noise_scale in NOISE_SCALE:
                        if signal == "circle":
                            for radius in RADII:
                                for magnitude in MAGNITUDES:
                                    threshold = THR_CIRCLE
                                    if threshold >= magnitude:
                                        continue
    
                                    print(
                                        f"Running: signal={signal}, signal_fwhm={signal_fwhm}, "
                                        f"noise_fwhm={noise_fwhm}, alpha={alpha}, samples={n}, "
                                        f"radius={radius}, magnitude={magnitude},  noise scale = {noise_scale}"
                                    )
    
                                    out = numerical_test(
                                        repeat=REPEAT,
                                        alpha=alpha,
                                        signal_type=signal,
                                        threshold=threshold,
                                        signal_fwhm=signal_fwhm,
                                        noise_fwhm=noise_fwhm,
                                        samples=n,
                                        boot_rep=BOOT_REP,
                                        noise_scale=noise_scale,
                                        radius=radius,
                                        magnitude=magnitude,
                                    )
    
                                    row = {
                                        "signal_type": signal,
                                        "signal_fwhm": signal_fwhm,
                                        "noise_fwhm": noise_fwhm,
                                        "alpha": alpha,
                                        "sample": n,
                                        "threshold": threshold,
                                        "threshold_frac": np.nan,
                                        "boot_rep": BOOT_REP,
                                        "repeat": REPEAT,
                                        "noise_scale": NOISE_SCALE,
                                        "radius_param": radius,
                                        "magnitude": magnitude,
                                        "maxsig": np.nan,
                                        "n_valid": out["n_valid"],
                                        "coverage_directed": out["coverage_directed"],
                                        "coverage_symmetric": out["coverage_symmetric"],
                                        "d_true_to_est_mean": out["d_true_to_est_mean"],
                                        "d_true_to_est_sd": out["d_true_to_est_sd"],
                                        "d_est_to_true_mean": out["d_est_to_true_mean"],
                                        "d_est_to_true_sd": out["d_est_to_true_sd"],
                                        "h_sym_mean": out["h_sym_mean"],
                                        "h_sym_sd": out["h_sym_sd"],
                                        "rel_asym_mean": out["rel_asym_mean"],
                                        "rel_asym_sd": out["rel_asym_sd"],
                                        "runtime_mean": out["runtime_mean"],
                                        "runtime_sd": out["runtime_sd"],
                                    }
                                    results.append(row)
                                    print(out["coverage_directed"], out["coverage_symmetric"])
                        elif signal == "grad":
                            for maxsig in MAXSIGS:
                                for threshold_frac in THRESHOLD_FRACS:
                                    threshold = threshold_frac * maxsig
                                    if threshold <= MINSIG or threshold >= maxsig:
                                        continue
    
                                    print(
                                        f"Running: signal={signal}, signal_fwhm={signal_fwhm}, "
                                        f"noise_fwhm={noise_fwhm}, alpha={alpha}, samples={n}, "
                                        f"maxsig={maxsig}, thr_frac={threshold_frac}"
                                    )
    
                                    out = numerical_test(
                                        repeat=REPEAT,
                                        alpha=alpha,
                                        signal_type=signal,
                                        threshold=threshold,
                                        signal_fwhm=signal_fwhm,
                                        noise_fwhm=noise_fwhm,
                                        samples=n,
                                        boot_rep=BOOT_REP,
                                        noise_scale=noise_scale,
                                        maxsig=maxsig,
                                    )
    
                                    row = {
                                        "signal_type": signal,
                                        "signal_fwhm": signal_fwhm,
                                        "noise_fwhm": noise_fwhm,
                                        "alpha": alpha,
                                        "sample": n,
                                        "threshold": threshold,
                                        "threshold_frac": threshold_frac,
                                        "boot_rep": BOOT_REP,
                                        "repeat": REPEAT,
                                        "noise_scale": NOISE_SCALE,
                                        "radius_param": np.nan,
                                        "magnitude": np.nan,
                                        "maxsig": maxsig,
                                        "n_valid": out["n_valid"],
                                        "coverage_directed": out["coverage_directed"],
                                        "coverage_symmetric": out["coverage_symmetric"],
                                        "d_true_to_est_mean": out["d_true_to_est_mean"],
                                        "d_true_to_est_sd": out["d_true_to_est_sd"],
                                        "d_est_to_true_mean": out["d_est_to_true_mean"],
                                        "d_est_to_true_sd": out["d_est_to_true_sd"],
                                        "h_sym_mean": out["h_sym_mean"],
                                        "h_sym_sd": out["h_sym_sd"],
                                        "rel_asym_mean": out["rel_asym_mean"],
                                        "rel_asym_sd": out["rel_asym_sd"],
                                        "runtime_mean": out["runtime_mean"],
                                        "runtime_sd": out["runtime_sd"],
                                    }
                                    results.append(row)
                                    print(out["coverage_directed"], out["coverage_symmetric"])
df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
out_csv = "results/numerical_hausdorff_asymptotic_direct.csv"
df.to_csv(out_csv, index=False)
print(f"Saved results to {out_csv}")

import matplotlib.pyplot as plt

signal_type = "circle"      # or "grad"
signal_fwhm = 3
noise_fwhm = 3
noise_scale = 2
n_subject = 50
thr = 2

radius = 30
magnitude = 3
maxsig = 3

# make one signal
signal = make_signal(
    signal_type=signal_type,
    signal_fwhm=signal_fwhm,
    radius=radius,
    magnitude=magnitude,
    maxsig=maxsig,
)

# simulate one dataset
instances = simulate_instances(
    signal=signal,
    n_subject=n_subject,
    noise_fwhm=noise_fwhm,
    noise_scale=noise_scale,
)

# sample mean field
coefficient = np.mean(instances, axis=0)

# interpolated contour points
true_bdry, _, _, _ = contour_points(signal, thr)
est_bdry, _, _, _ = contour_points(coefficient, thr)

# choose one subject to display
one_instance = instances[0]


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1) true signal
axes[0].imshow(signal, origin="lower", cmap="viridis")
if len(true_bdry) > 0:
    axes[0].plot(true_bdry[:, 1], true_bdry[:, 0], "r.", ms=1)
axes[0].set_title("True signal")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# 2) one simulated subject
axes[1].imshow(one_instance, origin="lower", cmap="viridis")
if len(true_bdry) > 0:
    axes[1].plot(true_bdry[:, 1], true_bdry[:, 0], "r.", ms=1)
axes[1].set_title("One simulated instance")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# 3) sample mean with estimated contour
axes[2].imshow(coefficient, origin="lower", cmap="viridis")
if len(true_bdry) > 0:
    axes[2].plot(true_bdry[:, 1], true_bdry[:, 0], "r.", ms=1, label="true boundary")
if len(est_bdry) > 0:
    axes[2].plot(est_bdry[:, 1], est_bdry[:, 0], "w.", ms=1, label="estimated boundary")
axes[2].set_title("Sample mean")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.show()

"""