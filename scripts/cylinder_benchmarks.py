import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------
# 1) Initial PCA-based estimate (same as before)
# --------------------------------------------------------
def initial_cylinder(xyz):
    centroid = xyz.mean(axis=0)
    X = xyz - centroid
    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, np.argmax(eigvals)]
    axis_dir /= np.linalg.norm(axis_dir)

    # project onto axis
    proj = X @ axis_dir
    h = proj.max() - proj.min()

    # bottom point
    bottom = centroid + axis_dir * proj.min()

    # radius
    perp = X - np.outer(proj, axis_dir)
    r = np.median(np.linalg.norm(perp, axis=1))

    # spherical angles
    ux, uy, uz = axis_dir
    polar = np.arccos(uz)
    azim = np.arctan2(uy, ux) % (2*np.pi)

    return np.array([bottom[0], bottom[1], bottom[2], r, h, polar, azim])


# --------------------------------------------------------
# 2) Cylinder residual function
# --------------------------------------------------------
def cylinder_residuals(params, xyz):
    x_c, y_c, z_c, r, h, polar, azim = params

    # axis direction
    u = np.array([
        np.sin(polar) * np.cos(azim),
        np.sin(polar) * np.sin(azim),
        np.cos(polar)
    ])

    p = np.array([x_c, y_c, z_c])  # bottom point

    # vector from bottom point to each point
    V = xyz - p

    # projection onto axis
    t = V @ u

    # closest point on axis
    C = p + np.outer(t, u)

    # radial distance - radius
    d = np.linalg.norm(xyz - C, axis=1)

    return d - r


# --------------------------------------------------------
# 3) Refinement wrapper
# --------------------------------------------------------
def refined_cylinder_fit(xyz, initial = None):
    # initial estimate
    if initial is None:
        initial = initial_cylinder(xyz)

    # optimize
    res = least_squares(
        cylinder_residuals,
        initial,
        args=(xyz,),
        method='lm',
        max_nfev=200,
        xtol=1e-10,
        ftol=1e-10
    )

    return res.x  # [x_c, y_c, z_c, r, h, polar, azim]

import numpy as np
from scipy.optimize import least_squares

# --------------------------------------------------------
# Axis from spherical angles
# --------------------------------------------------------
def axis_from_angles(polar, azim):
    return np.array([
        np.sin(polar) * np.cos(azim),
        np.sin(polar) * np.sin(azim),
        np.cos(polar)
    ])

# --------------------------------------------------------
# Initial estimate WITHOUT height inside params
# --------------------------------------------------------
def initial_cylinder_params_no_h(xyz):
    centroid = xyz.mean(axis=0)
    X = xyz - centroid

    cov = X.T @ X
    _, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, -1]
    axis_dir /= np.linalg.norm(axis_dir)

    # spherical angles
    ux, uy, uz = axis_dir
    polar = np.arccos(uz)
    azim  = np.arctan2(uy, ux) % (2*np.pi)

    # rough bottom point using min projection
    proj = X @ axis_dir
    bottom = centroid + axis_dir * proj.min()

    # rough radius
    perp = X - np.outer(proj, axis_dir)
    r = np.median(np.linalg.norm(perp, axis=1))

    return np.array([bottom[0], bottom[1], bottom[2], r, polar, azim])

# --------------------------------------------------------
# Residuals for optimization (NO HEIGHT TERM)
# --------------------------------------------------------
def cylinder_residuals_no_h(params, xyz):
    x_c, y_c, z_c, r, polar, azim = params
    p = np.array([x_c, y_c, z_c])
    u = axis_from_angles(polar, azim)

    V = xyz - p
    t = V @ u
    C = p + np.outer(t, u)
    d = np.linalg.norm(xyz - C, axis=1)

    return d - r

# --------------------------------------------------------
# Final refined cylinder fit
# --------------------------------------------------------
def refined_cylinder_fit_stable(xyz):
    # 1) initial guess (no height)
    p0 = initial_cylinder_params_no_h(xyz)

    # 2) Optimize stable parameters
    res = least_squares(
        cylinder_residuals_no_h, p0,
        args=(xyz,), method="lm",
        max_nfev=200, xtol=1e-12, ftol=1e-12
    )
    x_c, y_c, z_c, r, polar, azim = res.x

    # 3) After optimizing axis + bottom, now compute height robustly
    p = np.array([x_c, y_c, z_c])
    u = axis_from_angles(polar, azim)

    proj = (xyz - p) @ u
    # robust height with percentiles
    low, high = np.percentile(proj, [5, 95])
    h = high - low

    # re-anchor bottom → exactly low percentile
    bottom = p + u * low

    # return in requested format
    return np.array([
        bottom[0], bottom[1], bottom[2],   # x_c, y_c, z_c (fixed!)
        r,
        h,
        polar, azim
    ])