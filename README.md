# lorenz-score-sde  
A minimal, type-annotated PyTorch reference implementation of score-based generative modeling on the Lorenz-63 attractor in $\mathbb{R}^3$.

A reference pipeline for:

- Data generation via forward SDE simulation.
- Training a denoising score network.
- Sampling with Predictor–Corrector SDE solvers.
- Evaluating sample fidelity through marginal metrics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

```bash
# 1. Clone and enter the repo
git clone https://github.com/sbdeery/lorenz-score-sde.git
cd lorenz-score-sde

# 2. Create an isolated environment (recommended)
python -m venv .venv            # or `conda create -n lorenzscore python=3.11`
source .venv/bin/activate       # on Windows: .venv\Scripts\activate

# 3. Upgrade pip and install runtime requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) install developer tools—linters, Black, MyPy, etc.
pip install -r requirements-dev.txt
```

## Usage

Step by step:

```bash
# Data
python -m data.make_data --dist lorenz --outdir data

# Train
python -m scripts.train --data data/lorenz_train_norm.npy

# Sample
python -m scripts.sample --ckpt outputs/ckpts/e50.pt --n 150000 --sample_type pc --outfile outputs/pc_samples.npz

# Evaluate
python -m scripts.eval_marginals --data data/lorenz_train_norm.npy --samples outputs/pc_samples.npz --stats data/lorenz_stats.json
```

## Examples

View the minimal demo: `examples/minimal_demo.ipynb`

## Development

```bash
black .
isort .
flake8 .
mypy .
```

## Citation

```bibtex
@inproceedings{song2021score,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Ermon, Stefano},
  booktitle={ICLR},
  year={2021}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Maintainer: Sebastian Deery  
Email: sbdeery@uchicago.edu  
