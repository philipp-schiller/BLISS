import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# -----------------------
# Helper: covariance ellipse
# -----------------------
def draw_cov_ellipse(ax, x, y, cov, scale=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = scale * np.sqrt(vals)

    ell = Ellipse((x, y), width, height, angle=angle, **kwargs)
    ax.add_patch(ell)

# -----------------------
# Primitive
# -----------------------
N_curve = 600
t = np.linspace(0, 2*np.pi, N_curve)

a, b = 3.0, 2.0
x = a * np.cos(t)
y = b * np.sin(t)

# -----------------------
# Sensor
# -----------------------
sensor = np.array([-5.0, -5.0])

# -----------------------
# Visibility-based density
# -----------------------
# compute angle for each point
angles = np.zeros_like(t)
visibility_vals = np.zeros_like(t)

for i in range(len(t)):
    p = np.array([x[i], y[i]])

    ray = p - sensor
    dist = np.linalg.norm(ray)
    ray_dir = ray / dist

    # angle of ray
    angles[i] = np.arctan2(ray_dir[1], ray_dir[0])

    # visibility check
    normal = np.array([x[i]/(a**2), y[i]/(b**2)])
    normal /= np.linalg.norm(normal)

    visibility = np.dot(normal, -ray_dir)
    visibility_vals[i] = visibility

# unwrap angles to avoid discontinuities
angles = np.unwrap(angles)

# compute dθ/dt
dtheta_dt = np.gradient(angles, t)

# density ∝ |dθ/dt|
rho = np.abs(dtheta_dt)

# apply visibility mask
rho[visibility_vals <= 0] = 0.0

# normalize
rho /= np.trapz(rho, t)


# -----------------------
# Find visible region boundaries (tangent points)
# -----------------------
visible_mask = visibility_vals > 0
indices = np.where(visible_mask)[0]

start_idx = indices[0]
end_idx = indices[-1]

p_start = np.array([x[start_idx], y[start_idx]])
p_end   = np.array([x[end_idx], y[end_idx]])

# -----------------------
# Sampling
# -----------------------
N_samples = 12

cdf = np.cumsum(rho)
cdf /= cdf[-1]

rand_vals = np.random.rand(N_samples)
t_samples = np.interp(rand_vals, cdf, t)

xs = a * np.cos(t_samples)
ys = b * np.sin(t_samples)

# -----------------------
# Laser-dependent covariance
# -----------------------
def compute_cov(p):
    # ray
    ray = p - sensor
    dist = np.linalg.norm(ray)
    ray_dir = ray / dist

    # surface normal (ellipse)
    normal = np.array([p[0] / (a**2), p[1] / (b**2)])
    normal /= np.linalg.norm(normal)

    # incidence angle (angle between normal and incoming ray)
    cos_theta = np.dot(normal, -ray_dir)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    # convert to angle behavior
    grazing = 1 - cos_theta   # 0 = orthogonal, 1 = grazing

    # distance scaling
    base = 0.02 + 0.05 * dist

    # NEW: strong anisotropy variation
    sigma_parallel = base * (1.0 + 2.5 * grazing)     # grows strongly at grazing
    sigma_perp     = base * (0.5 + 0.5 * grazing)     # also grows, but less

    # directions
    direction = ray_dir
    perp = np.array([-direction[1], direction[0]])

    cov = (
        sigma_parallel**2 * np.outer(direction, direction) +
        sigma_perp**2     * np.outer(perp, perp)
    )

    return cov

# noisy observations
noisy_pts = []

for xi, yi in zip(xs, ys):
    p = np.array([xi, yi])
    cov = compute_cov(p)
    q = np.random.multivariate_normal(p, cov)
    noisy_pts.append(q)

noisy_pts = np.array(noisy_pts)

# -----------------------
# Plot helper: draw sensor + visibility
# -----------------------
def draw_sensor_and_visibility(ax):
    ax.scatter(sensor[0], sensor[1], color='red', s=40)

    # tangent rays
    ax.plot([sensor[0], p_start[0]],
            [sensor[1], p_start[1]],
            color='orange', linewidth=1.5)

    ax.plot([sensor[0], p_end[0]],
            [sensor[1], p_end[1]],
            color='orange', linewidth=1.5)

    # highlight visible arc
    #ax.plot(x[visible_mask], y[visible_mask],
    #        color='orange', linewidth=3)

# -----------------------
# Plot
# -----------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Panel 1: Primitive
axes[0].plot(x, y, color='black', linewidth=2)
draw_sensor_and_visibility(axes[0])
axes[0].set_title(r"Geometric Primitive")
axes[0].axis('equal')
axes[0].axis('off')

# Panel 2: Density
sc = axes[1].scatter(x, y, c=rho, cmap='viridis', s=5)
draw_sensor_and_visibility(axes[1])
axes[1].set_title(r"Source density $\pi$")
axes[1].axis('equal')
axes[1].axis('off')

cbar = plt.colorbar(sc, ax=axes[1], fraction=0.046)
cbar.set_label("density")

# Panel 3: Covariance field

axes[2].plot(x, y, color='lightgray')
draw_sensor_and_visibility(axes[2])

for i in range(start_idx, end_idx, 20):
    p = np.array([x[i], y[i]])
    cov = compute_cov(p)

    # ray
    axes[2].plot([sensor[0], p[0]],
                 [sensor[1], p[1]],
                 color='gray', linewidth=0.5, alpha=0.3)

    draw_cov_ellipse(axes[2],
                     p[0], p[1],
                     cov,
                     edgecolor='red',
                     facecolor='none',
                     linewidth=1)

axes[2].set_title(r"Source-dependent noise model $\eta$")
axes[2].axis('equal')
axes[2].axis('off')

# Panel 4: Samples + observations
axes[3].plot(x, y, color='lightgray')
draw_sensor_and_visibility(axes[3])

axes[3].scatter(xs, ys, color='blue', zorder=3)
axes[3].scatter(noisy_pts[:,0], noisy_pts[:,1],
                color='black', zorder=3)

for i in range(len(xs)):
    axes[3].plot([xs[i], noisy_pts[i,0]],
                 [ys[i], noisy_pts[i,1]],
                 color='gray', linewidth=1)

axes[3].set_title(r"Corresponding measurement")
axes[3].axis('equal')
axes[3].axis('off')

plt.tight_layout()
plt.show()