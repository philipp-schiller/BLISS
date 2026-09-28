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
# Helper: bottom title
# ============================================================
def add_bottom_title(ax, text, fontsize=11, y=-0.08):
    ax.text(
        0.5, y, text,
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=fontsize
    )

# ============================================================
# Primitive
# ============================================================
N_curve = 600
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
np.random.seed(0)  # for reproducibility

N_samples = 12
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

# ============================================================
# Noisy observations
# ============================================================
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

# Optional normalization:
# density /= np.max(density)

# ============================================================
# Common panel styling
# IMPORTANT: same limits + equal aspect in ALL panels
# so the primitive ellipse stays the same size everywhere
# ============================================================
XLIM = (-5.8, 4.2)
YLIM = (-5.8, 4.2)

def style_panel(ax):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_anchor('C')
    ax.axis('off')

def draw_sensor(ax):
    ax.scatter(sensor[0], sensor[1], color='red', s=40, zorder=5)
    ax.plot([sensor[0], p_start[0]], [sensor[1], p_start[1]], color='orange', zorder=2)
    ax.plot([sensor[0], p_end[0]], [sensor[1], p_end[1]], color='orange', zorder=2)

def save_fig(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches='tight')
    fig.savefig(f"{name}.svg", bbox_inches='tight')
    fig.savefig(f"{name}.png", dpi=300, bbox_inches='tight')

# ============================================================
# Detached right-side colorbar helper
# These colorbars do NOT affect subplot geometry
# ============================================================
def add_right_colorbar(fig, ref_ax, mappable, x0=0.91, width=0.015,
                       height_frac=0.80, label=None):
    """
    Add a vertical colorbar in the right margin, aligned to ref_ax.

    Parameters
    ----------
    fig : Figure
    ref_ax : Axes
        The panel whose vertical position is used for alignment.
    mappable : matplotlib mappable
        Scatter or image returned by scatter()/imshow().
    x0 : float
        Left edge of the colorbar in figure coordinates.
    width : float
        Width of the colorbar in figure coordinates.
    height_frac : float
        Fraction of ref_ax height used by the colorbar.
    """
    pos = ref_ax.get_position()

    cbar_height = pos.height * height_frac
    cbar_y0 = pos.y0 + 0.5 * (pos.height - cbar_height)

    cax = fig.add_axes([x0, cbar_y0, width, cbar_height])
    cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=9)

    if label is not None:
        cbar.set_label(label, fontsize=10)

    return cbar

# ============================================================
# Panel drawing functions
# ============================================================
def plot_panel_1(ax):
    ax.plot(x, y, color='black', linewidth=2)
    draw_sensor(ax)
    style_panel(ax)
    add_bottom_title(ax, r"Geometric primitive", fontsize=15, y=-0.06)

def plot_panel_2(ax):
    sc = ax.scatter(x, y, c=rho, cmap='viridis', s=8)
    draw_sensor(ax)
    style_panel(ax)
    add_bottom_title(ax, r"Source density $\pi$", fontsize=15, y=-0.06)
    return sc

def plot_panel_3(ax):
    ax.plot(x, y, color='lightgray', linewidth=1.5)
    draw_sensor(ax)

    for i in range(start_idx, end_idx, 20):
        p = np.array([x[i], y[i]])
        cov = compute_cov(p)

        ax.plot([sensor[0], p[0]], [sensor[1], p[1]],
                color='gray', alpha=0.3, linewidth=0.8)

        draw_cov_ellipse(ax, p[0], p[1], cov,
                         edgecolor='red', facecolor='none', linewidth=1.2)

    style_panel(ax)
    add_bottom_title(ax, r"Source-dependent noise model $\eta$", fontsize=15, y=-0.06)

def plot_panel_4(ax):
    ax.plot(x, y, color='lightgray', linewidth=1.5)
    draw_sensor(ax)

    # true source points
    ax.scatter(xs, ys, color='blue', s=28, zorder=4)

    # uncomment these if you want the noisy points instead / as well
    ax.scatter(noisy_pts[:, 0], noisy_pts[:, 1], color='black', s=15, zorder=5)

    # optional: connect source to noisy observation
    # for i in range(len(xs)):
    #     ax.plot([xs[i], noisy_pts[i, 0]],
    #             [ys[i], noisy_pts[i, 1]],
    #             color='gray', linewidth=0.8)

    style_panel(ax)
    add_bottom_title(ax, r"Corresponding measurement", fontsize=15, y=-0.06)

def plot_panel_5(ax):
    im = ax.imshow(
        density,
        extent=[-4, 4, -4, 4],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )

    ax.plot(x, y, color='white', linewidth=1.5)
    draw_sensor(ax)

    style_panel(ax)
    add_bottom_title(ax, r"Marginal likelihood $p(x)$", fontsize=15, y=-0.04)
    return im

# ============================================================
# Combined multi-panel figure
# Layout:
#   1    2    4    5
#   1    3    4    5
# ============================================================
fig, axs = plt.subplot_mosaic(
    [
        ["1", "2", "5", "4"],
        ["1", "3", "5", "4"]
    ],
    figsize=(13, 7.5),
    constrained_layout=False,
    gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]}
)

# Leave room on the right for detached colorbars
fig.subplots_adjust(
    left=0.03,
    right=0.88,
    top=0.96,
    bottom=0.08,
    wspace=0.18,
    hspace=0.18
)

# Draw panels
plot_panel_1(axs["1"])
sc2 = plot_panel_2(axs["2"])
plot_panel_3(axs["3"])
plot_panel_4(axs["4"])
im5 = plot_panel_5(axs["5"])

# Finalize layout before reading axis positions
fig.canvas.draw()

# Right-side detached colorbars
# Same x0 keeps them vertically aligned on the right-hand side
add_right_colorbar(
    fig, axs["2"], sc2,
    x0=0.905, width=0.015,
    height_frac=0.72
)

add_right_colorbar(
    fig, axs["5"], im5,
    x0=0.905, width=0.015,
    height_frac=0.80
)

save_fig(fig, "combined_figure_custom_layout")
plt.show()
