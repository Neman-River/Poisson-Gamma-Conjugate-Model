import numpy as np
import streamlit as st
from scipy.stats import gamma, nbinom
import plotly.graph_objects as go

st.set_page_config(page_title="Poisson-Gamma Bayesian Inference", layout="wide")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Extra breathing room above every section banner */
.step-banner {
    padding: 14px 22px;
    border-radius: 8px;
    margin: 56px 0 4px 0;
    color: white;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
/* Thin accent line beneath each banner */
.step-rule {
    border: none;
    border-top: 3px solid;
    margin: 0 0 20px 0;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


def section(number: str, title: str, color: str) -> None:
    """Render a coloured section banner + matching accent rule."""
    st.markdown(
        f'<div class="step-banner" style="background:{color};">'
        f'{number} &nbsp;·&nbsp; {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<hr class="step-rule" style="border-color:{color};">',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Prior Hyperparameters")
alpha_prior = st.sidebar.slider("α (alpha)", 0.1, 20.0, 1.0, 0.1)
beta_prior = st.sidebar.slider("β (beta)", 0.01, 5.0, 0.1, 0.01)

prior_mean = alpha_prior / beta_prior
prior_std = np.sqrt(alpha_prior) / beta_prior
st.sidebar.markdown(f"**E[λ] = α/β** = {prior_mean:.2f}")
st.sidebar.markdown(f"**SD[λ] = √α/β** = {prior_std:.2f}")

st.sidebar.divider()
st.sidebar.header("Data")
data_mode = st.sidebar.radio("Data source", ["Simulate", "Enter manually"])

if data_mode == "Simulate":
    n_obs = st.sidebar.slider("N (observations)", 1, 100, 10)
    true_lambda = st.sidebar.slider("True λ", 0.5, 30.0, 5.0, 0.5)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    rng = np.random.default_rng(int(seed))
    observations = rng.poisson(true_lambda, size=n_obs).tolist()
else:
    raw = st.sidebar.text_area("Counts (comma-separated)", "3,5,4,6,5,7,4,5,6,4")
    try:
        observations = [int(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError:
        st.sidebar.error("Enter integers separated by commas.")
        observations = []

# ── Derived quantities ────────────────────────────────────────────────────────
x = np.array(observations, dtype=float)
N = len(x)
sum_x = x.sum() if N > 0 else 0.0

alpha_post = alpha_prior + sum_x
beta_post = beta_prior + N

# ── Page title ────────────────────────────────────────────────────────────────
st.title("Poisson-Gamma Conjugate Model")
st.markdown(
    "Bayesian inference for a Poisson rate λ using the Gamma conjugate prior. "
    "Adjust the sliders on the left to explore how prior beliefs and observed data "
    "shape the posterior."
)

# ── Process overview diagram ──────────────────────────────────────────────────
section("0", "Process Overview", "#555E6B")
st.markdown(
    "The diagram below shows the full Bayesian workflow — from prior belief, "
    "through observed data, to the posterior and prediction."
)
st.graphviz_chart("""
digraph {
    rankdir=LR
    graph [bgcolor=transparent splines=ortho nodesep=0.6]
    node  [fontname="Helvetica" fontsize=13 style="filled,rounded" shape=box
           margin="0.25,0.15"]
    edge  [fontname="Helvetica" fontsize=11 color="#555555"]

    Prior [label="Prior\nGamma(α, β)"
           fillcolor="#AED6F1" color="#2E86C1"]

    Lambda [label="Latent rate  λ"
            fillcolor="#D5F5E3" color="#1E8449" shape=ellipse]

    Data [label="Observed counts\nx₁, x₂, …, xₙ"
          fillcolor="#E8DAEF" color="#7D3C98"]

    Likelihood [label="Likelihood\nPoisson(λ)"
                fillcolor="#FDEBD0" color="#CA6F1E"]

    Update [label="Conjugate update\nα* = α + Σxᵢ\nβ* = β + N"
            fillcolor="#FDFEFE" color="#717D7E"]

    Posterior [label="Posterior\nGamma(α*, β*)"
               fillcolor="#FADBD8" color="#C0392B"]

    Predictive [label="Posterior predictive\nNegBin(α*, β*/(β*+1))"
                fillcolor="#FDEBD0" color="#D35400"]

    Prior    -> Lambda    [label=" encodes belief\n about λ"]
    Lambda   -> Likelihood
    Likelihood -> Data   [label=" generates"]
    Prior    -> Update
    Data     -> Update
    Update   -> Posterior [label=" yields"]
    Posterior -> Predictive [label=" marginalise\n over λ"]
}
""")

# ── Section 1 – Prior ─────────────────────────────────────────────────────────
section("1", "Prior Distribution", "#2E86C1")
st.markdown("We encode our belief about the rate λ before seeing any data:")
st.latex(r"\lambda \sim \text{Gamma}(\alpha,\, \beta)")
st.markdown(
    "The Gamma distribution is parameterised here with **shape α** and **rate β**, "
    "so the mean is α/β."
)

max_x_prior = max(gamma.ppf(0.999, alpha_prior, scale=1 / beta_prior), 1.0)
lam_grid = np.linspace(0, max_x_prior, 300)
prior_pdf = gamma.pdf(lam_grid, a=alpha_prior, scale=1 / beta_prior)

fig_prior = go.Figure()
fig_prior.add_trace(
    go.Scatter(
        x=lam_grid, y=prior_pdf, mode="lines",
        line=dict(color="steelblue", width=2),
        name="Prior PDF",
    )
)
fig_prior.update_layout(
    xaxis_title="λ", yaxis_title="Density",
    title=f"Gamma({alpha_prior:.1f}, {beta_prior:.2f}) Prior",
    height=300, margin=dict(t=40, b=40),
)
st.plotly_chart(fig_prior, use_container_width=True)

prior_ci_lo = gamma.ppf(0.025, alpha_prior, scale=1 / beta_prior)
prior_ci_hi = gamma.ppf(0.975, alpha_prior, scale=1 / beta_prior)
c1, c2, c3 = st.columns(3)
c1.metric("Mean (α/β)", f"{prior_mean:.3f}")
c2.metric("Std (√α/β)", f"{prior_std:.3f}")
c3.metric("95 % CI", f"[{prior_ci_lo:.2f}, {prior_ci_hi:.2f}]")

# ── Section 2 – Observed Data ─────────────────────────────────────────────────
section("2", "Observed Data", "#7D3C98")
st.markdown(
    "Each observation xᵢ is a **count** — the number of events that occurred "
    "during one fixed time window (e.g. visitors to a coffee shop on one day). "
    "Counts like these are modelled with a **Poisson likelihood**:"
)
st.latex(r"x_i \mid \lambda \;\sim\; \text{Poisson}(\lambda), \quad i = 1, \dots, N")
st.markdown(
    "Given the true rate λ, the probability of seeing exactly k events in one window is:"
)
st.latex(r"P(X = k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}")

if data_mode == "Simulate":
    st.markdown(
        f"**Simulated data:** {N} draws from Poisson(λ = {true_lambda}) using seed {int(seed)}. "
        "This mimics repeatedly recording the count over N independent time windows."
    )
else:
    st.markdown(
        "**Manual data:** counts you entered directly, treated as N independent "
        "Poisson observations with the same unknown rate λ."
    )

if N == 0:
    st.warning("No data yet — enter or simulate some observations.")
else:
    counts = np.arange(int(x.max()) + 1)
    freq = np.array([(x == k).sum() for k in counts])

    fig_data = go.Figure()
    fig_data.add_trace(
        go.Bar(x=counts, y=freq, marker_color="mediumpurple", name="Count frequency")
    )
    fig_data.update_layout(
        xaxis_title="Count", yaxis_title="Frequency",
        title="Distribution of Observed Counts",
        height=280, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_data, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.metric("N (observations)", N)
    d2.metric("Σxᵢ", int(sum_x))
    d3.metric("Sample mean", f"{x.mean():.3f}")

# ── Section 3 – Bayesian Update ───────────────────────────────────────────────
section("3", "Bayesian Update", "#C0392B")
st.markdown(
    "The Gamma prior is **conjugate** to the Poisson likelihood, so the posterior "
    "is also a Gamma distribution. The update simply accumulates the observed counts "
    "into the hyperparameters:"
)
st.latex(r"\alpha^* = \alpha + \sum x_i")
st.latex(r"\beta^* = \beta + N")
st.latex(r"\lambda \mid \mathbf{x} \sim \text{Gamma}(\alpha^*,\, \beta^*)")

# Overlay prior vs posterior
max_x_post = max(
    gamma.ppf(0.999, alpha_post, scale=1 / beta_post),
    gamma.ppf(0.999, alpha_prior, scale=1 / beta_prior),
    1.0,
)
lam_grid2 = np.linspace(0, max_x_post, 400)
prior_pdf2 = gamma.pdf(lam_grid2, a=alpha_prior, scale=1 / beta_prior)
post_pdf = gamma.pdf(lam_grid2, a=alpha_post, scale=1 / beta_post)

fig_update = go.Figure()
fig_update.add_trace(
    go.Scatter(
        x=lam_grid2, y=prior_pdf2, mode="lines",
        line=dict(color="steelblue", width=2, dash="dash"),
        name="Prior",
    )
)
fig_update.add_trace(
    go.Scatter(
        x=lam_grid2, y=post_pdf, mode="lines",
        line=dict(color="crimson", width=2),
        name="Posterior",
    )
)
fig_update.update_layout(
    xaxis_title="λ", yaxis_title="Density",
    title="Prior vs Posterior",
    height=320, margin=dict(t=40, b=40),
    legend=dict(x=0.75, y=0.95),
)
st.plotly_chart(fig_update, use_container_width=True)

post_mean = alpha_post / beta_post
post_median = gamma.ppf(0.5, alpha_post, scale=1 / beta_post)
post_ci_lo = gamma.ppf(0.025, alpha_post, scale=1 / beta_post)
post_ci_hi = gamma.ppf(0.975, alpha_post, scale=1 / beta_post)

p1, p2, p3 = st.columns(3)
p1.metric("Posterior mean", f"{post_mean:.3f}")
p2.metric("Posterior median", f"{post_median:.3f}")
p3.metric("95 % credible interval", f"[{post_ci_lo:.2f}, {post_ci_hi:.2f}]")

# ── Section 4 – Posterior Predictive ─────────────────────────────────────────
section("4", "Posterior Predictive Distribution", "#D35400")
st.markdown(
    "Integrating out λ over the posterior gives the **Negative Binomial** distribution "
    "as the predictive distribution for a new observation x̃:"
)
st.latex(
    r"\tilde{x} \sim \text{NegBin}\!\left(r = \alpha^*,\; p = \frac{\beta^*}{\beta^* + 1}\right)"
)
st.markdown(
    "Here *r* is the number of successes (shape) and *p* is the success probability "
    "in the scipy parameterisation."
)

r_nb = alpha_post
p_nb = beta_post / (beta_post + 1)

# Range: cover ~99.9 % of the mass
k_max = int(nbinom.ppf(0.999, r_nb, p_nb)) + 1
k_vals = np.arange(0, k_max + 1)
pmf_vals = nbinom.pmf(k_vals, r_nb, p_nb)

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Bar(
        x=k_vals, y=pmf_vals,
        marker_color="darkorange",
        name="NegBin PMF",
    )
)
fig_pred.update_layout(
    xaxis_title="Future count x̃", yaxis_title="Probability",
    title="Posterior Predictive (Negative Binomial)",
    height=300, margin=dict(t=40, b=40),
)
st.plotly_chart(fig_pred, use_container_width=True)

pred_mean = nbinom.mean(r_nb, p_nb)
pred_std = nbinom.std(r_nb, p_nb)
q1, q2 = st.columns(2)
q1.metric("Predictive mean", f"{pred_mean:.3f}")
q2.metric("Predictive std", f"{pred_std:.3f}")
