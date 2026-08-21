# Lorenz-63 Score-Based Generative Modeling

A compact PyTorch research implementation of score-based generative modeling on the Lorenz-63 attractor. The repository was developed for experiments at the intersection of generative modeling and chaotic dynamical systems.

## What is included

- Lorenz-63 simulation, whitening, and data-generation utilities
- A time-conditioned MLP score model trained with VP-SDE score matching
- Predictor-Corrector, Euler-Maruyama, and probability-flow ODE samplers
- Marginal-fidelity diagnostics using KS, Wasserstein-1, and KDE-based comparisons
- A minimal end-to-end notebook and precomputed demo outputs

## Installation

```bash
git clone https://github.com/sbdeery/lorenz63-sgm.git
cd lorenz63-sgm

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

A portable Conda environment is also provided in `environment.yml`.

## End-to-end example

Generate and whiten a Lorenz-63 training set:

```bash
python -m data.make_data --dist lorenz --outdir data
```

Train the score model:

```bash
python -m scripts.train \
  --data data/lorenz_train_norm.npy \
  --epochs 50 \
  --out outputs
```

Draw Predictor-Corrector samples:

```bash
python -m scripts.sample \
  --ckpt outputs/ckpts/e50.pt \
  --n 150000 \
  --sample_type pc \
  --outfile outputs/pc_lorenz_samples.npz
```

Evaluate generated samples against the Lorenz-63 reference distribution:

```bash
python -m scripts.eval_marginals \
  --data data/lorenz_train_norm.npy \
  --samples outputs/pc_lorenz_samples.npz \
  --stats data/lorenz_stats.json \
  --outdir outputs/marginals
```

## Demo and development

`examples/minimal_demo.ipynb` walks through the pipeline. `demo_outputs/` contains a small set of precomputed outputs for inspection without rerunning the full experiment.

For development tools:

```bash
pip install -e ".[dev]"
black .
isort .
flake8 .
mypy .
```

## Scope

This is a research codebase for low-dimensional chaotic-system experiments, not a general-purpose generative-modeling library. The implementation is intentionally small enough to expose the training, sampling, and diagnostic steps directly.

## License

MIT License. See `LICENSE`.

## Contact

Sebastian Deery — sbdeery@gmail.com
