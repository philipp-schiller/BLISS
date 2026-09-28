import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fontsize = 28

# ============================================================
# Helper: covariance ellipse
# ============================================================
def draw_cov_ellipse(ax, x, y, cov, scale=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = scale * np.sqrt(vals)

    ell = Ellipse((x, y), width, height, angle=angle, **kwargs)
    ax.add_patch(ell)


# ============================================================
# Helper: title below panel
# ============================================================
def add_bottom_title(ax, text, fontsize=11, y=-0.08):
    ax.text(
        0.5, y, text,
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=fontsize
    )


# ============================================================
# Helper: horizontal colorbar above panel
# ============================================================
def add_top_colorbar(fig, ax, mappable, label=None):
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation='horizontal',
        pad=0.02,
        fraction=0.05,
        location='top'
    )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=15)

    if label is not None:
        cbar.set_label(label, fontsize=15)

    return cbar

def add_side_colorbar(fig, ax, mappable, label=None):
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation='vertical',
        pad=0.02,
        fraction=0.05,
        location='right'
    )
    #cbar.ax.xaxis.set_ticks_position('default')
    #cbar.ax.xaxis.set_label_position('default')
    cbar.ax.tick_params(labelsize=15)

    if label is not None:
        cbar.set_label(label, fontsize=15)

    return cbar
# ============================================================
# Primitive
# ============================================================
N_curve = 800
t = np.linspace(0, 2*np.pi, N_curve)

a, b = 3.0, 2.0
x = a * np.cos(t)
y = b * np.sin(t)

# ============================================================
# Sensor
# ============================================================
sensor = np.array([-4.0, -4.0])

# ============================================================
# Visibility + density
# ============================================================
angles = np.zeros_like(t)
visibility_vals = np.zeros_like(t)

for i in range(len(t)):
    p = np.array([x[i], y[i]])

    ray = p - sensor
    dist = np.linalg.norm(ray)
    ray_dir = ray / dist

    angles[i] = np.arctan2(ray_dir[1], ray_dir[0])

    normal = np.array([p[0] / (a**2), p[1] / (b**2)])
    normal /= np.linalg.norm(normal)

    visibility_vals[i] = np.dot(normal, -ray_dir)

angles = np.unwrap(angles)
dtheta_dt = np.gradient(angles, t)

rho = np.abs(dtheta_dt)
rho[visibility_vals <= 0] = 0.0
rho /= np.trapz(rho, t)

visible_mask = visibility_vals > 0
indices = np.where(visible_mask)[0]
start_idx, end_idx = indices[0], indices[-1]

p_start = np.array([x[start_idx], y[start_idx]])
p_end   = np.array([x[end_idx], y[end_idx]])

# ============================================================
# Sampling
# ============================================================
np.random.seed(69)

N_samples = 50
cdf = np.cumsum(rho)
cdf /= cdf[-1]

t_samples = np.interp(np.random.rand(N_samples), cdf, t)
xs = a * np.cos(t_samples)
ys = b * np.sin(t_samples)

# ============================================================
# Noise model
# ============================================================
def compute_cov(p):
    ray = p - sensor
    dist = np.linalg.norm(ray)
    ray_dir = ray / dist

    normal = np.array([p[0] / (a**2), p[1] / (b**2)])
    normal /= np.linalg.norm(normal)

    cos_theta = np.clip(np.dot(normal, -ray_dir), 0, 1)
    grazing = 1 - cos_theta

    base = 0.02 + 0.05 * dist

    sigma_parallel = base * (1 + 2.5 * grazing)
    sigma_perp     = base * (0.5 + 0.5 * grazing)

    d = ray_dir
    pvec = np.array([-d[1], d[0]])

    return (
        sigma_parallel**2 * np.outer(d, d) +
        sigma_perp**2     * np.outer(pvec, pvec)
    )

# noisy observations
noisy_pts = []
for xi, yi in zip(xs, ys):
    p = np.array([xi, yi])
    noisy_pts.append(np.random.multivariate_normal(p, compute_cov(p)))
noisy_pts = np.array(noisy_pts)

# ============================================================
# Marginal likelihood
# ============================================================
grid_x, grid_y = np.meshgrid(
    np.linspace(-4, 4, 200),
    np.linspace(-4, 4, 200)
)

grid_points = np.stack([grid_x, grid_y], axis=-1)
density = np.zeros(grid_x.shape)

for i in range(start_idx, end_idx):
    if rho[i] == 0:
        continue

    s = np.array([x[i], y[i]])
    cov = compute_cov(s)

    diff = grid_points - s
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    norm_const = 1.0 / (2 * np.pi * np.sqrt(det_cov))
    exponent = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)

    density += rho[i] * norm_const * np.exp(-0.5 * exponent)

# Optional normalization if desired:
density /= np.max(density)

# ============================================================
# Helper
# ============================================================
def draw_sensor(ax):
    ax.scatter(sensor[0], sensor[1], color='black', marker="x", s=60, zorder=5, linewidth=4)
    ax.plot([sensor[0], p_start[0]], [sensor[1], p_start[1]], color='orange', zorder=2)
    ax.plot([sensor[0], p_end[0]], [sensor[1], p_end[1]], color='orange', zorder=2)

    for i in range(start_idx, end_idx, 10):
        p = np.array([x[i], y[i]])

        ax.plot([sensor[0], p[0]], [sensor[1], p[1]],
                color='gray', alpha=0.3, linewidth=0.8)


def save_fig(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.svg", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=300, bbox_inches='tight')


# ============================================================
# PANEL 1: Primitive
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(x, y, color='black', linewidth=3)
draw_sensor(ax)

ax.axis('equal')
ax.axis('off')
#add_bottom_title(ax, r"Geometric primitive $\theta$\n+ measuring device", fontsize=fontsize, y=-0.06)
add_bottom_title(ax, "Geometric primitive $\\theta$\n(+ measuring device)", fontsize=fontsize, y=-0.06)

save_fig(fig, "panel_1_primitive")


# ============================================================
# PANEL 2: Density
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
sc = ax.scatter(x, y, c=rho, cmap='Reds', s=20)
#ax.plot(x, y, color='black', linewidth=1)
draw_sensor(ax)

ax.axis('equal')
ax.axis('off')
add_bottom_title(ax, r"Source density $\pi$", fontsize=fontsize, y=-0.06)
add_side_colorbar(fig, ax, sc)

save_fig(fig, "panel_2_density")


# ============================================================
# PANEL 3: Noise model
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(x, y, color='black', linewidth=3)
draw_sensor(ax)

for i in range(start_idx, end_idx, 10):
    p = np.array([x[i], y[i]])
    cov = compute_cov(p)

    #ax.plot([sensor[0], p[0]], [sensor[1], p[1]],
    #        color='gray', alpha=0.3, linewidth=0.8)

    draw_cov_ellipse(ax, p[0], p[1], cov,
                     edgecolor='red', facecolor='none', linewidth=1.2)

ax.axis('equal')
ax.axis('off')
add_bottom_title(ax, r"Noise model $\eta$", fontsize=fontsize, y=-0.06)

save_fig(fig, "panel_3_noise")


# ============================================================
# PANEL 4: Samples
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
ax.plot(x, y, color='lightgray', linewidth=1.5)
draw_sensor(ax)

#ax.scatter(xs, ys, color='blue', s=28, zorder=4)

# Uncomment if you want noisy observations shown:
ax.scatter(noisy_pts[:, 0], noisy_pts[:, 1], color='black', s=20, zorder=5)
# for i in range(len(xs)):
#     ax.plot([xs[i], noisy_pts[i, 0]],
#             [ys[i], noisy_pts[i, 1]],
#             color='gray', linewidth=0.8)

ax.axis('equal')
ax.axis('off')
add_bottom_title(ax, r"Corresponding measurement", fontsize=fontsize, y=-0.06)

save_fig(fig, "panel_4_samples")


# ============================================================
# PANEL 5: Marginal likelihood
# ============================================================
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

im = ax.imshow(
    density,
    extent=[-4, 4, -4, 4],
    origin='lower',
    cmap="Reds"
)

ax.plot(x, y, color='black', linewidth=3)
draw_sensor(ax)

ax.axis('equal')
ax.axis('off')
add_bottom_title(ax, r"Observation model $p(\, \cdot  \mid \theta)$", fontsize=25, y=-0.06)
add_top_colorbar(fig, ax, im)

save_fig(fig, "panel_5_marginal")

plt.show()
