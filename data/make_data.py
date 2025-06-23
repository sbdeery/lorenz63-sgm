
"""Generate Lorenz-63 or sphere dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t, xyz):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x, y, z = xyz
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def gen_lorenz(n_seeds=1, dt=0.01, t_end=5000.0):
    rng = np.random.default_rng()
    seeds = rng.uniform(-5, 5, size=(n_seeds, 3))
    traj = []
    for s in seeds:
        _ = solve_ivp(lorenz, (-500, 0), s, max_step=dt)
        sol = solve_ivp(lorenz, (0, t_end), _.y[:, -1], t_eval=np.arange(0, t_end, dt))
        traj.append(sol.y.T)
    return np.concatenate(traj, axis=0)


def gen_sphere(n=300_000, radius=10.0):
    rng = np.random.default_rng()
    u, theta, phi = rng.uniform(0, 1, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, np.pi, n)
    r = radius * u ** (1 / 3)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def whiten(arr):
    mu = arr.mean(axis=0)
    cov = np.cov(arr.T)
    U, S, _ = np.linalg.svd(cov)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    return (arr - mu) @ inv_sqrt, mu, inv_sqrt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dist", choices=["lorenz", "sphere"], required=True)
    p.add_argument("--outdir", default="data")
    args = p.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dist == "lorenz":
        raw = gen_lorenz()
    else:
        raw = gen_sphere()
    data, mu, inv = whiten(raw.astype(np.float32))
    np.save(out / f"{args.dist}_train_norm.npy", data)
    json.dump({"mu": mu.tolist(), "inv": inv.tolist()}, open(out / f"{args.dist}_stats.json", "w"))
    print("Saved", data.shape, "to", out)


if __name__ == "__main__":
    main()
