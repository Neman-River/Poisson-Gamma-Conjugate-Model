# Poisson-Gamma Bayesian Inference

An interactive Streamlit app for exploring Bayesian inference of a Poisson rate λ using the Gamma conjugate prior.

## What it does

The app walks through the full Bayesian workflow in four sections:

1. **Prior** — set your belief about λ via Gamma(α, β) hyperparameters
2. **Data** — simulate Poisson counts or enter your own
3. **Bayesian Update** — conjugate update yields a new Gamma posterior
4. **Posterior Predictive** — marginalising over λ gives a Negative Binomial distribution for future counts

## Run locally

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/Poisson-Gamma-Conjugate-Model.git
cd poisson-gamma-dist
uv run streamlit run app.py
```

## Dependencies

Managed via `pyproject.toml` and `uv.lock`:

- `numpy`
- `scipy`
- `plotly`
- `streamlit`

## Project structure

```
.
├── app.py                    # Streamlit app
├── PoissonGammaPython.ipynb  # Concept notebook
├── pyproject.toml
└── uv.lock
```
