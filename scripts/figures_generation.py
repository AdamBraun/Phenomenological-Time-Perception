"""
Generate demonstration figures for the multiscale ε–δ time-perception framework.

Outputs
-------
1. epsilon_delta_ladder_v3.png
2. cantor_partition_depth5_v3.png
3. psychometric_curve_v3.png
4. algorithm1_schematic_v3.png
5. three simulation figures:
   ─ temporal_bisection.png
   ─ vierordt_bias.png
   ─ weber_variance.png
All files are written to the current working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from scipy.special import expit  # logistic sigmoid
from scipy.interpolate import UnivariateSpline

# Model parameters
epsilon = 0.1  # s
delta = 0.3    # s

# ----------------------------------------------------------------------
# Section 1 – ε–δ ladder (single tier)
# ----------------------------------------------------------------------
def plot_epsilon_delta_ladder(eps=0.1, delta=0.3, k_max=6,
                              fname="epsilon_delta_ladder_v3.png"):
    xmax = delta * (k_max + 0.2)
    x = np.linspace(0, xmax, 1000)
    T = np.piecewise(
        x,
        [x < eps,
         (x >= eps) & (x <= delta),
         x > delta],
        [0, 1, lambda z: np.ceil(z / delta)]
    )
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axvspan(0, eps, color="0.90", label="sub-ε (imperceptible)")
    ax.axvspan(eps, delta, color="0.80", label="ε–δ (single-tick)")
    ax.step(x, T, where="post", lw=1.5, label="T(d)")
    ax.axvline(eps, ls="--", lw=1)
    ax.axvline(delta, ls="--", lw=1)
    ax.set_xlabel(r"metric distance $d$")
    ax.set_ylabel(r"perceived ticks $T_{\varepsilon,\delta}(d)$")
    ax.set_ylim(-0.2, k_max + 1)
    ax.set_xlim(0, xmax)
    ax.set_title(rf"ε–δ ladder ($\varepsilon={eps}$, $\delta={delta}$)")
    ax.legend(frameon=False, loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------
# Section 2 – Cantor-like partition
# ----------------------------------------------------------------------
def plot_cantor_partition(depth=5, fname="cantor_partition_depth5_v3.png"):
    fig, ax = plt.subplots(figsize=(6, 3))
    height = 0.14
    segments = [(0.0, 1.0)]
    y = 0
    grey = np.linspace(0.85, 0.25, depth + 1)
    for d in range(depth + 1):
        for a, b in segments:
            ax.add_patch(Rectangle((a, y), b - a, height,
                                   facecolor=str(grey[d]), edgecolor="none"))
        new = []
        for a, b in segments:
            third = (b - a) / 3
            new.extend([(a, a + third), (b - third, b)])
        segments = new
        ax.text(-0.03, y + height / 2, f"{d}", ha="right", va="center", fontsize=8)
        y += height + 0.03
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, y)
    ax.set_xlabel(r"$\Sigma$ (normalised coordinate)")
    ax.set_ylabel("Recursion depth")
    ax.set_yticks([])
    ax.set_title(f"Cantor-like Partition of Σ (depth = {depth})")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------
# Section 3 – Psychometric grid
# ----------------------------------------------------------------------
def plot_psychometric(eps=0.1, delta=0.3,
                      durations=(0.2, 0.4, 0.8, 1.6),
                      disps=(0.05, 0.10, 0.20, 0.40),
                      fname="psychometric_curve_v3.png"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for disp in disps:
        k_eps = eps * (0.10 / disp)
        k_delta = delta * (0.10 / disp)
        ticks = np.piecewise(
            durations,
            [np.array(durations) < k_eps,
             (np.array(durations) >= k_eps) & (np.array(durations) <= k_delta),
             np.array(durations) > k_delta],
            [0, 1, lambda z: np.ceil(z / k_delta)]
        )
        ax.plot(durations, ticks, marker="o", label=f"d={disp:.2f}°")
    ref = np.linspace(min(durations), max(durations), 200)
    ax.plot(ref, ref / delta, ls="--", lw=0.8, color="grey",
            label="physical / δ")
    ax.set_xscale("log")
    ax.set_xlabel("Physical duration (s)")
    ax.set_ylabel("Predicted perceived units")
    ax.set_title("Predicted Psychometric Functions (4×4 grid)")
    ax.legend(title="Displacement", frameon=False, fontsize="small")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------
# Section 4 – Algorithm 1 schematic
# ----------------------------------------------------------------------
def plot_algorithm_schematic(fname="algorithm1_schematic_v3.png"):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    def box(txt, xy, w=1.6, h=0.55):
        x, y = xy
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.15", lw=1)
        ax.add_patch(rect)
        ax.text(x, y, txt, ha="center", va="center", fontsize=9)
    xs = [0, 2.4, 4.8]
    ys = [0, -1.2]
    box(r"$s_1^{(0)},\,s_2^{(0)}$", (xs[0], ys[0]))
    box(r"$d^{(0)}$",            (xs[1], ys[0]))
    box(r"$\mathbf{T}^{(0)}$",   (xs[2], ys[0]))
    for i in range(2):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.9, ys[0]),
                                     (xs[i+1] - 0.9, ys[0]),
                                     arrowstyle="->", mutation_scale=10))
    ax.add_patch(FancyArrowPatch((xs[2], ys[0]-0.28),
                                 (xs[0], ys[1]+0.28),
                                 arrowstyle="->", mutation_scale=10))
    ax.text(xs[0] + 1.2, ys[0]-0.5, r"Map to $s^{(n+1)}$", fontsize=8)
    box(r"$s_1^{(1)},\,s_2^{(1)}$", (xs[0], ys[1]))
    box(r"$d^{(1)}$",            (xs[1], ys[1]))
    box(r"$\mathbf{T}^{(1)}$",   (xs[2], ys[1]))
    for i in range(2):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.9, ys[1]),
                                     (xs[i+1] - 0.9, ys[1]),
                                     arrowstyle="->", mutation_scale=10))
    ax.text(xs[1], ys[1]-0.6, r"$\vdots$", ha="center", va="center", fontsize=12)
    ax.text(-1.0, ys[0], "n = 0", va="center", fontsize=8)
    ax.text(-1.0, ys[1], "n = 1", va="center", fontsize=8)
    ax.set_xlim(-1.5, 6)
    ax.set_ylim(-2, 0.8)
    ax.set_title("Recursive Multiscale Tick Computation (Algorithm 1)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------
# Section 5 – Synthetic psychophysics
# ----------------------------------------------------------------------
def T_single(d, eps, delta):
    d = np.asarray(d)
    t = np.zeros_like(d, dtype=float)
    t[d < eps] = 0
    band = (d >= eps) & (d <= delta)
    t[band] = 1
    above = d > delta
    t[above] = np.ceil(d[above] / delta)
    return t

def simulate_behaviour(eps=0.1, delta=0.3,
                       durations=(0.2, 0.4, 0.8, 1.6),
                       n_trials=500, noise_sd=0.15,
                       out_prefix=""):
    rng = np.random.default_rng(0)
    dur_vec = []
    perc_ticks = []
    for d in durations:
        base = T_single(d, eps, delta)
        noisy = base + rng.normal(0, noise_sd, n_trials)
        noisy = np.clip(noisy, 0, None)
        dur_vec.append(np.full(n_trials, d))
        perc_ticks.append(noisy)
    dur_vec = np.concatenate(dur_vec)
    perc_ticks = np.concatenate(perc_ticks)
    # Temporal bisection
    crit = 0.5 * (T_single(0.4, eps, delta) + T_single(1.6, eps, delta))
    prob_long = [np.mean(perc_ticks[dur_vec == d] >= crit) for d in durations]
    plt.figure()
    plt.plot(durations, prob_long, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Physical duration (s)")
    plt.ylabel("P('long')")
    plt.title("Temporal bisection – model")
    plt.tight_layout()
    plt.savefig(out_prefix + "temporal_bisection.png", dpi=300)
    plt.close()
    # Vierordt bias
    reproductions = perc_ticks * delta
    bias = [reproductions[dur_vec == d].mean() / d - 1 for d in durations]
    plt.figure()
    plt.axhline(0, ls="--", lw=0.8)
    plt.plot(durations, bias, marker="o")
    plt.xlabel("Physical duration (s)")
    plt.ylabel("Relative bias")
    plt.title("Vierordt bias – model")
    plt.tight_layout()
    plt.savefig(out_prefix + "vierordt_bias.png", dpi=300)
    plt.close()
    # Weber variance
    means = [reproductions[dur_vec == d].mean() for d in durations]
    sds = [reproductions[dur_vec == d].std() for d in durations]
    plt.figure()
    plt.plot(means, sds, marker="o")
    plt.xlabel("Mean reproduced duration (s)")
    plt.ylabel("SD reproduced duration (s)")
    plt.title("Weber variance – model")
    plt.tight_layout()
    plt.savefig(out_prefix + "weber_variance.png", dpi=300)
    plt.close()

def T_eps_delta(d, eps=epsilon, delt=delta):
    """Single-tier ε–δ mapping for scalar physical duration d (seconds)."""
    d = np.asarray(d)
    return np.where(d < eps, 0,
                   np.where(d <= delt, 1, d / delt))  # Continuous for d > δ

# Figure 5 — Temporal Bisection Psychometric (synthetic predictions, smoothed)
def fig_temporal_bisection(short_ref=0.4, long_ref=1.6, n_probe=200, sigma=0.3, filename="temporal_bisection.png"):
    probes = np.linspace(short_ref, long_ref, n_probe)

    # Perceived ticks
    T_short = T_eps_delta(short_ref)
    T_long = T_eps_delta(long_ref)
    T_mid = 0.5 * (T_short + T_long)

    T_probe = T_eps_delta(probes)

    # Psychometric function: P("long") with wider logistic
    p_long = expit((T_probe - T_mid) / sigma)

    plt.figure(figsize=(6, 4))
    plt.plot(probes, p_long, 'b-', label="P(long)")  # Smooth line
    plt.axvline(short_ref, ls='--', lw=0.8, color='black', label='short ref')
    plt.axvline(long_ref, ls='--', lw=0.8, color='black', label='long ref')
    plt.axhline(0.5, ls=':', lw=0.8, color='gray', label='P = 0.5')
    plt.xlabel("Probe duration (s)")
    plt.ylabel('P("long" response)')
    plt.title("Synthetic Temporal Bisection\nModel midpoint at P=0.5")
    plt.legend(frameon=False, fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Figure 6 — Vierordt Bias Curve (synthetic predictions, smoothed)
def fig_vierordt_bias(phys=np.linspace(0.2, 2.0, 300), k_central=0.4, filename="vierordt_bias.png"):
    # True perceived ticks
    T_true = T_eps_delta(phys)
    mean_T = T_true.mean()
    T_reprod = T_true + k_central * (mean_T - T_true)

    # Convert reproduced ticks back to seconds
    d_reprod = np.where(T_reprod < 0.5, 0, T_reprod * delta)

    # Spline smoothing to preserve bow shape
    spl = UnivariateSpline(phys, d_reprod, s=0.05)  # Small s for minimal distortion
    d_reprod_smooth = spl(phys)

    plt.figure(figsize=(6, 4))
    plt.plot(phys, d_reprod_smooth, 'b-', lw=2, label="reproduced")  # Smooth line
    plt.plot(phys, phys, ls='--', color='gray', label="veridical")
    plt.xlabel("Physical duration (s)")
    plt.ylabel("Reproduced duration (s)")
    plt.title("Synthetic Vierordt Curve\n(central-tendency bias k=0.4)")
    plt.legend(frameon=False, fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Figure 7 — Apparatus Diagram
def fig_apparatus_diagram(filename="apparatus_diagram.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')

    # monitor frame
    ax.add_patch(Rectangle((-3, 0), 6, 2, fill=False, linewidth=2))
    ax.text(0, 1, "OLED 144 Hz 24″", ha='center', va='center', fontsize=8)

    # stimulus window
    ax.add_patch(Rectangle((-1.5, 0.25), 3, 1.5, color='black'))
    ax.text(0, 1, "Random-dot\nfield", color='white',
            ha='center', va='center', fontsize=8)

    # chin-rest
    ax.add_patch(Rectangle((-0.8, -0.4), 1.6, 0.6, fill=False, linewidth=1.5))
    ax.text(0, -0.1, "Chin-rest", ha='center', va='top', fontsize=8)

    # dashed distance markers
    ax.plot([-3, -4.2], [1.5, 1.5], ls='--', lw=1, color='gray')
    ax.plot([ 3,  4.2], [1.5, 1.5], ls='--', lw=1, color='gray')
    ax.text(-4.3, 1.5, "57 cm", rotation=90, va='center',
            ha='right', fontsize=8)

    # horizontal brace (optional)
    arrow = FancyArrowPatch((3, 2.15), (-3, 2.15),
                            arrowstyle='<->', mutation_scale=10, lw=1)
    ax.add_patch(arrow)
    ax.text(0, 2.25, "≈ 53 cm", ha='center', va='bottom', fontsize=7)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()

# ----------------------------------------------------------------------
# Master execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    plot_epsilon_delta_ladder()
    plot_cantor_partition()
    plot_psychometric()
    plot_algorithm_schematic()
    simulate_behaviour()
    fig_temporal_bisection()
    fig_vierordt_bias()
    fig_apparatus_diagram()
