# lorenz-score-sde

<!-- CI badge will be added here after you configure GitHub Actions -->
<!-- Coverage badge will be added here after you integrate Codecov -->

A minimal, type-annotated reference implementation of score-based generative modeling for the Lorenz‑63 attractor in ℝ³. This repository demonstrates:

- Data generation of invariant-measure samples via forward SDE simulation.
- Training a denoising score network through denoising score matching.
- Sampling using Predictor–Corrector SDE solvers.
- Evaluation of sample fidelity via marginal distribution metrics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

Clone the repository and install runtime dependencies:

```bash
git clone https://github.com/sbdeery/lorenz-score-modeling.git
cd lorenz-score-modeling
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install developer tools for formatting, linting, and type checking:

```bash
pip install -r requirements-dev.txt
```

## Usage

### Data Generation

```bash
python -m data.make_data --dist lorenz --outdir data
```

### Training

```bash
python -m scripts.train   --data data/lorenz_train_norm.npy   --outdir outputs   --epochs 100   --batch-size 512
```

### Sampling

```bash
python -m scripts.sample   --ckpt outputs/ckpts/e50.pt   --n 150000   --sample_type pc   --out outputs/pc_samples.npz
```

### Evaluation

```bash
python -m scripts.eval_marginals   --data data/lorenz_train_norm.npy   --samples outputs/pc_samples.npz   --stats data/lorenz_stats.json   --outdir outputs/marginals   --bins 150
```

## Examples

See `examples/minimal_demo.ipynb` for a complete walkthrough of data generation, training, sampling, and evaluation.

## Development

- **Formatting & Linting:**
  ```bash
  black .
  isort .
  flake8 .
  mypy .
  ```
- **Testing:**
  ```bash
  pytest --cov
  ```
- **Continuous Integration:** See `.github/workflows/ci.yml`.

## Citation

If you use this code, please cite:

Song, Yang and Ermon, Stefano. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Maintainer: Sebastian Deery  
Email: sbdeery@uchicago.edu
