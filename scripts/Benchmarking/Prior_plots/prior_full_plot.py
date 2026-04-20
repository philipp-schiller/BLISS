import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------------------------------------
# 1. Priors on w
# ---------------------------------------------------------
mu_w = 0.4
sigma2_weak_w = 0.002602
sigma2_strong_w = 0.00065

sigma_weak_w = np.sqrt(sigma2_weak_w)
sigma_strong_w = np.sqrt(sigma2_strong_w)

# Tight x-range around the mean
x_w = np.linspace(mu_w - 3*sigma_weak_w, mu_w + 3*sigma_weak_w, 800)

p_weak_w = norm.pdf(x_w, loc=mu_w, scale=sigma_weak_w)
p_strong_w = norm.pdf(x_w, loc=mu_w, scale=sigma_strong_w)

# ---------------------------------------------------------
# 2. Noise priors
# ---------------------------------------------------------
mu_n = 0.0225
sigma2_weak_n = 0.00001
sigma2_strong_n = 0.000001

sigma_weak_n = np.sqrt(sigma2_weak_n)
sigma_strong_n = np.sqrt(sigma2_strong_n)

xmin_n = max(0, mu_n - 3*sigma_weak_n)
xmax_n = mu_n + 3*sigma_weak_n
x_n = np.linspace(xmin_n, xmax_n, 800)

p_weak_n = norm.pdf(x_n, loc=mu_n, scale=sigma_weak_n)
p_strong_n = norm.pdf(x_n, loc=mu_n, scale=sigma_strong_n)

# ---------------------------------------------------------
# Combined Plot
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---------------- Left: Source priors on w ----------------
ax = axes[0]
ax.plot(x_w, p_weak_w, label="Weak prior", linewidth=2)
ax.plot(x_w, p_strong_w, "--", label="Strong prior", linewidth=2)
ax.axvline(mu_w, color="black", linestyle=":", linewidth=2, label="Mean $\\mu_w$")

ax.set_xlabel(r"$w$", fontsize=25)
ax.set_ylabel(r"$p(w)$", fontsize=25)
ax.tick_params(axis='both', labelsize=20)
ax.grid(alpha=0.2)
ax.legend(fontsize=18)

# ---------------- Right: Noise priors ----------------
ax = axes[1]
ax.plot(x_n, p_weak_n, label="Weak prior", linewidth=2)
ax.plot(x_n, p_strong_n, "--", label="Strong prior", linewidth=2)
ax.axvline(mu_n, color="black", linestyle=":", linewidth=2, label="Mean $\\mu_{\\sigma^2}$")

ax.set_xlabel(r"$\sigma^2$", fontsize=25)
ax.set_ylabel(r"$p(\sigma^2)$", fontsize=25)
ax.tick_params(axis='both', labelsize=20)
ax.grid(alpha=0.2)
ax.legend(fontsize=18, loc = 1)

plt.tight_layout()

# Save as vector graphic
plt.savefig("combined_priors.png", dpi=300, bbox_inches="tight")

plt.show()