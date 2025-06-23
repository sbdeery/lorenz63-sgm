
# Lorenz Score Project

Minimal, type-annotated reference implementation of score‑based generative modelling
for the Lorenz‑63 attractor and a uniform sphere in **ℝ³**.

## Quickstart

```bash
# create dataset
python -m data.make_data --dist lorenz --outdir data

# train
python -m scripts.train --data data/lorenz_train_norm.npy --out outputs

# sample some points
python -m scripts.sample --ckpt outputs/ckpts/e50.pt --n 150000 --sample_type pc

# visualise
python -m scripts.plot_samples --train data/lorenz_train_norm.npy --samples outputs/pc_lorenz_samples.npz --stats data/lorenz_stats.json

# Evaluate
python -m scripts.eval_marginals --data data/lorenz_train_norm.npy --samples outputs/pc_lorenz_samples.npz --stats data/lorenz_stats.json --outdir outputs/marginals --bins 150
```