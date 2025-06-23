"""Generate Lorenz-63 or sphere dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]


def lorenz(t: float, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Lorenz-63 vector field.
    Args:
        t: time (unused).
        xyz: state array of shape (3,).
    Returns:
        Array of shape (3,) giving dx/dt.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    x, y, z = xyz
    return np.array(
        [sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64
    )


def gen_lorenz(
    n_seeds: int = 1,
    dt: float = 0.01,
    t_end: float = 5000.0,
) -> NDArray[np.float64]:
    """
    Simulate the Lorenz attractor from random seeds.
    Args:
        n_seeds: number of initial seeds.
        dt: time step for simulation.
        t_end: final time.
    Returns:
        Array of shape (n_seeds * (t_end/dt), 3) containing the trajectory.
    """
    rng = np.random.default_rng()
    seeds = rng.uniform(-5, 5, size=(n_seeds, 3))
    traj: list[NDArray[np.float64]] = []
    for s in seeds:
        # spin-up
        warm = solve_ivp(lorenz, (-500, 0), s, max_step=dt)
        sol = solve_ivp(
            lorenz,
            (0, t_end),
            warm.y[:, -1],
            t_eval=np.arange(0, t_end, dt),
            max_step=dt,
        )
        # cast so mypy knows sol.y.T is NDArray[np.float64]
        traj.append(cast(NDArray[np.float64], sol.y.T))
    return np.concatenate(traj, axis=0)


def gen_sphere(
    n: int = 300_000,
    radius: float = 10.0,
) -> NDArray[np.float64]:
    """
    Sample uniformly from the volume of a sphere.
    Args:
        n: number of points.
        radius: sphere radius.
    Returns:
        Array of shape (n, 3) with x,y,z coordinates.
    """
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, np.pi, n)
    r = radius * u ** (1 / 3)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def whiten(
    arr: NDArray[np.float32],
) -> Tuple[
    NDArray[np.float32],  # whitened data
    NDArray[np.float32],  # mean
    NDArray[np.float64],  # inverse sqrt-covariance
]:
    """
    Whiten the data to zero mean, unit covariance.
    Args:
        arr: input data of shape (N,3), dtype float32.
    Returns:
        whitened arr, mean, and inverse sqrt covariance.
    """
    mu = arr.mean(axis=0)
    cov = np.cov(arr.T)
    U, S, _ = np.linalg.svd(cov)
    inv_sqrt = U @ np.diag(1.0 / np.sqrt(S)) @ U.T
    white = (arr - mu) @ inv_sqrt
    return white, mu, inv_sqrt


def main() -> None:
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
    json.dump(
        {"mu": mu.tolist(), "inv": inv.tolist()},
        open(out / f"{args.dist}_stats.json", "w"),
    )
    print("Saved", data.shape, "to", out)


if __name__ == "__main__":
    main()
