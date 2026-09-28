"""
Generalized Cylinder shape class.

A generalized cylinder is a surface swept by a cross-section of radius r(t)
along a 3D spine curve c(t), both represented as B-splines.

The surface is parameterized as:
    S(t, θ) = c(t) + r(t) * (cos(θ) * N(t) + sin(θ) * B(t))

where:
    - c(t) is the spine curve (B-spline in R^3)
    - r(t) is the radius function (B-spline in R)
    - N(t), B(t) are the normal and binormal of the rotation-minimizing frame
    - t ∈ [0, 1] is arc-length-like parameter along the spine
    - θ ∈ [0, 2π) is the angular parameter around the cross-section

The shape parameters θ (confusingly same letter, different meaning from the
angular parameter — we use 'params' for the shape parameters throughout) are
laid out as:

    params = [c_control_pts (n_spine * 3), r_control_pts (n_radius)]

The rotation-minimizing frame (Bishop frame) is propagated along the spine
from an initial frame orientation, avoiding the twisting artifacts of the
Frenet frame.
"""

from dataclasses import dataclass, field
from typing import Callable
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import minimize

from .base_shape import Shape, GammaFn, JacobianFn


# ---------------------------------------------------------------------------
# Helper: B-spline basis evaluation
# ---------------------------------------------------------------------------

def _bspline_basis(t: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate B-spline basis functions at parameter values t.

    Parameters
    ----------
    t : np.ndarray, shape (M,)
        Parameter values.
    knots : np.ndarray
        Knot vector (clamped), length = n_control_points + degree + 1.
    degree : int
        Polynomial degree (3 = cubic).

    Returns
    -------
    B : np.ndarray, shape (M, n_control_points)
        B[i, j] = B_j(t_i), the j-th basis function evaluated at t_i.
    """
    n_ctrl = len(knots) - degree - 1
    B = np.zeros((len(t), n_ctrl))
    for j in range(n_ctrl):
        # Use scipy BSpline with a unit coefficient vector to get basis function j
        coeffs = np.zeros(n_ctrl)
        coeffs[j] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=True)
        B[:, j] = spl(t)
    return B


def _bspline_basis_deriv(t: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Evaluate first derivative of B-spline basis functions at parameter values t.

    Returns
    -------
    dB : np.ndarray, shape (M, n_control_points)
    """
    n_ctrl = len(knots) - degree - 1
    dB = np.zeros((len(t), n_ctrl))
    for j in range(n_ctrl):
        coeffs = np.zeros(n_ctrl)
        coeffs[j] = 1.0
        spl = BSpline(knots, coeffs, degree, extrapolate=True)
        dspl = spl.derivative()
        dB[:, j] = dspl(t)
    return dB


def _make_clamped_knots(n_ctrl: int, degree: int, t_low: float = 0.0, t_high: float = 1.0) -> np.ndarray:
    """
    Build a clamped (open) uniform knot vector.

    A clamped knot vector has `degree+1` repeated knots at each end,
    ensuring the spline interpolates its first and last control points.

    Parameters
    ----------
    n_ctrl : int
        Number of control points.
    degree : int
        Polynomial degree.
    t_low, t_high : float
        Parameter domain.

    Returns
    -------
    knots : np.ndarray, length n_ctrl + degree + 1
    """
    n_internal = n_ctrl - degree - 1
    if n_internal <= 0:
        # No internal knots needed (e.g. cubic with 4 control points)
        internal = np.array([])
    else:
        internal = np.linspace(t_low, t_high, n_internal + 2)[1:-1]

    knots = np.concatenate([
        np.full(degree + 1, t_low),
        internal,
        np.full(degree + 1, t_high),
    ])
    return knots

from scipy.optimize import minimize

def fit_cylinder_ls(points: np.ndarray):
    pts = points

    def unpack(theta):
        x0, y0, z0, r, h, polar, azim = theta

        ux = np.sin(polar) * np.cos(azim)
        uy = np.sin(polar) * np.sin(azim)
        uz = np.cos(polar)

        axis = np.array([ux, uy, uz])
        axis /= np.linalg.norm(axis)

        bottom = np.array([x0, y0, z0])
        return bottom, axis, r, h

    def objective(theta):
        bottom, axis, r, h = unpack(theta)

        X = pts - bottom
        proj = X @ axis

        proj_clamped = np.clip(proj, 0.0, h)
        closest = bottom + np.outer(proj_clamped, axis)

        d = np.linalg.norm(pts - closest, axis=1)
        return np.sum((d - r) ** 2)

    # --- good initial guess via PCA ---
    centroid = pts.mean(axis=0)
    X = pts - centroid

    cov = X.T @ X
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    proj = X @ axis
    h0 = proj.max() - proj.min()
    bottom0 = centroid + axis * proj.min()

    perp = X - np.outer(proj, axis)
    r0 = np.median(np.linalg.norm(perp, axis=1))

    ux, uy, uz = axis / np.linalg.norm(axis)
    polar0 = np.arccos(uz)
    azim0 = np.arctan2(uy, ux)

    theta0 = np.array([*bottom0, r0, h0, polar0, azim0])

    bounds = [
        (None, None), (None, None), (None, None),
        (1e-3, None),
        (1e-3, None),
        (0, np.pi),
        (0, 2*np.pi)
    ]

    res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)

    return res.x



# ---------------------------------------------------------------------------
# Helper: Rotation-minimizing frame (Bishop frame)
# ---------------------------------------------------------------------------

def _rotation_minimizing_frame(
    positions: np.ndarray,
    tangents: np.ndarray,
    initial_normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate a rotation-minimizing (Bishop) frame along a curve.

    Uses the double-reflection method of Wang et al. (2008), which is
    efficient and avoids the undefined-at-inflection issues of the Frenet frame.

    Parameters
    ----------
    positions : np.ndarray, shape (M, 3)
        Points along the curve.
    tangents : np.ndarray, shape (M, 3)
        Unit tangent vectors at each point.
    initial_normal : Optional[np.ndarray], shape (3,)
        Initial normal vector. If None, an arbitrary normal perpendicular
        to tangents[0] is chosen automatically.

    Returns
    -------
    normals : np.ndarray, shape (M, 3)
        Normal vectors of the RMF.
    binormals : np.ndarray, shape (M, 3)
        Binormal vectors of the RMF (= tangent × normal).
    """
    M = len(positions)
    normals = np.zeros((M, 3))
    binormals = np.zeros((M, 3))

    # --- Initialize frame at t=0 ---
    T0 = tangents[0]
    if initial_normal is None:
        # Pick an arbitrary vector not parallel to T0
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(T0, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        N0 = ref - np.dot(ref, T0) * T0
        N0 /= np.linalg.norm(N0)
    else:
        N0 = initial_normal - np.dot(initial_normal, T0) * T0
        N0 /= np.linalg.norm(N0)

    normals[0] = N0
    binormals[0] = np.cross(T0, N0)

    # --- Propagate using double-reflection method ---
    for i in range(1, M):
        T_prev = tangents[i - 1]
        T_curr = tangents[i]
        N_prev = normals[i - 1]

        # Reflection 1: reflect N_prev across the plane bisecting T_prev and T_curr
        v1 = positions[i] - positions[i - 1]
        c1 = np.dot(v1, v1)
        if c1 < 1e-12:
            normals[i] = N_prev
            binormals[i] = np.cross(T_curr, N_prev)
            continue

        N_L = N_prev - (2.0 / c1) * np.dot(v1, N_prev) * v1
        T_L = T_prev - (2.0 / c1) * np.dot(v1, T_prev) * v1

        # Reflection 2: reflect N_L and T_L to align T_L with T_curr
        v2 = T_curr - T_L
        c2 = np.dot(v2, v2)
        if c2 < 1e-12:
            N_i = N_L
        else:
            N_i = N_L - (2.0 / c2) * np.dot(v2, N_L) * v2

        # Orthonormalize against T_curr for numerical stability
        N_i = N_i - np.dot(N_i, T_curr) * T_curr
        norm_N = np.linalg.norm(N_i)
        if norm_N < 1e-10:
            # Fallback: keep previous normal projected onto new tangent plane
            N_i = normals[i - 1] - np.dot(normals[i - 1], T_curr) * T_curr
            norm_N = np.linalg.norm(N_i)
        N_i /= norm_N

        normals[i] = N_i
        binormals[i] = np.cross(T_curr, N_i)

    return normals, binormals


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class GeneralizedCylinder(Shape):
    """
    Generalized cylinder with B-spline spine and radius function.

    Surface parametrization:
        S(t, θ) = c(t) + r(t) * (cos(θ) * N(t) + sin(θ) * B(t))

    where c(t) ∈ R^3 is the spine (B-spline), r(t) ∈ R^+ is the radius
    (B-spline), and {T(t), N(t), B(t)} is the rotation-minimizing frame.

    Shape parameters (the flat vector passed as `params`):
        [c_x_ctrl (n_spine,), c_y_ctrl (n_spine,), c_z_ctrl (n_spine,),
         r_ctrl (n_radius,)]

    Total: 3 * n_spine_ctrl + n_radius_ctrl parameters.

    Unlike the parent Shape class, GeneralizedCylinder does NOT require
    the user to pass parameter_dim, ambient_dim, or parametrization —
    these are determined automatically from the cylinder geometry.

    Parameters
    ----------
    n_spine_ctrl : int
        Number of B-spline control points for the spine curve c(t).
        Minimum: degree_spine + 1.
    n_radius_ctrl : int
        Number of B-spline control points for the radius function r(t).
        Minimum: degree_radius + 1.
    degree_spine : int, default=3
        Polynomial degree for the spine B-spline (cubic recommended).
    degree_radius : int, default=3
        Polynomial degree for the radius B-spline.
    n_frame_samples : int, default=100
        Number of points used when propagating the rotation-minimizing frame.
        Higher = more accurate frame, especially for highly curved spines.
    t_low : float, default=0.0
        Lower bound of the spine parameter domain.
    t_high : float, default=1.0
        Upper bound of the spine parameter domain.

    Examples
    --------
    >>> import numpy as np
    >>> cyl = GeneralizedCylinder(n_spine_ctrl=6, n_radius_ctrl=4)
    >>> params = np.zeros(cyl.get_n_params())
    >>> # Set spine to a straight line along z-axis
    >>> t_vals = np.linspace(0, 1, 6)
    >>> params[0*6:1*6] = 0.0          # x control points
    >>> params[1*6:2*6] = 0.0          # y control points
    >>> params[2*6:3*6] = t_vals       # z control points (0 to 1)
    >>> params[3*6:] = 0.3             # constant radius
    >>> t_grid = cyl.sample_parameter_space_grid([20, 32])
    >>> pts = cyl.evaluate(t_grid, params)  # (640, 3)
    """

    # Our own fields — declared first with defaults so they can be used
    # positionally or as keyword args without requiring the parent's fields.
    n_spine_ctrl: int = 6
    n_radius_ctrl: int = 4
    degree_spine: int = 3
    degree_radius: int = 3
    n_frame_samples: int = 100
    t_low: float = 0.0
    t_high: float = 1.0

    # Parent Shape fields — given sentinel defaults here; overwritten in
    # __post_init__ before super().__post_init__() is called.
    parameter_dim: int = field(default=2, init=False)
    ambient_dim: int = field(default=3, init=False)
    parametrization: GammaFn = field(default=None, init=False)
    jacobian_fn: Optional[JacobianFn] = field(default=None, init=False)
    domain_low: Union[float, Sequence[float]] = field(default=0.0, init=False)
    domain_high: Union[float, Sequence[float]] = field(default=1.0, init=False)
    periodic: Union[bool, Sequence[bool]] = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Build knot vectors and wire up the parametrization."""

        # Validate control point counts
        if self.n_spine_ctrl < self.degree_spine + 1:
            raise ValueError(
                f"n_spine_ctrl ({self.n_spine_ctrl}) must be >= degree_spine + 1 "
                f"({self.degree_spine + 1})"
            )
        if self.n_radius_ctrl < self.degree_radius + 1:
            raise ValueError(
                f"n_radius_ctrl ({self.n_radius_ctrl}) must be >= degree_radius + 1 "
                f"({self.degree_radius + 1})"
            )

        # Build clamped knot vectors (stored as plain instance attrs, not
        # dataclass fields, so they don't interfere with the parent)
        object.__setattr__(self, '_knots_spine', _make_clamped_knots(
            self.n_spine_ctrl, self.degree_spine, self.t_low, self.t_high
        ))
        object.__setattr__(self, '_knots_radius', _make_clamped_knots(
            self.n_radius_ctrl, self.degree_radius, self.t_low, self.t_high
        ))

        # Wire up parent Shape fields before calling super().__post_init__()
        object.__setattr__(self, 'parameter_dim', 2)
        object.__setattr__(self, 'ambient_dim', 3)
        object.__setattr__(self, 'domain_low', np.array([self.t_low, 0.0]))
        object.__setattr__(self, 'domain_high', np.array([self.t_high, 2.0 * np.pi]))
        object.__setattr__(self, 'periodic', np.array([False, True]))
        object.__setattr__(self, 'parametrization', self._parametrize)
        object.__setattr__(self, 'jacobian_fn', self._jacobian)

        super().__post_init__()

    # ------------------------------------------------------------------
    # Parameter layout helpers
    # ------------------------------------------------------------------

    def get_n_params(self) -> int:
        """Total number of shape parameters: 3*n_spine_ctrl + n_radius_ctrl."""
        return 3 * self.n_spine_ctrl + self.n_radius_ctrl

    def _unpack_params(self, params: np.ndarray):
        """
        Unpack the flat parameter vector into spine control points and radius
        control points.

        Returns
        -------
        cx : np.ndarray, shape (n_spine_ctrl,) — x control points of spine
        cy : np.ndarray, shape (n_spine_ctrl,)
        cz : np.ndarray, shape (n_spine_ctrl,)
        r_ctrl : np.ndarray, shape (n_radius_ctrl,)
        """
        ns = self.n_spine_ctrl
        cx = params[0 * ns: 1 * ns]
        cy = params[1 * ns: 2 * ns]
        cz = params[2 * ns: 3 * ns]
        r_ctrl = params[3 * ns: 3 * ns + self.n_radius_ctrl]
        return cx, cy, cz, r_ctrl

    # ------------------------------------------------------------------
    # Spine and radius evaluation
    # ------------------------------------------------------------------

    def eval_spine(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the spine curve c(t).

        Parameters
        ----------
        t : np.ndarray, shape (M,)
            Parameter values in [t_low, t_high].
        params : np.ndarray
            Shape parameters.

        Returns
        -------
        c : np.ndarray, shape (M, 3)
        """
        cx, cy, cz, _ = self._unpack_params(params)
        B = _bspline_basis(t, self._knots_spine, self.degree_spine)  # (M, n_spine)
        x = B @ cx
        y = B @ cy
        z = B @ cz
        return np.stack([x, y, z], axis=-1)  # (M, 3)

    def eval_spine_tangent(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the spine tangent dc/dt (unnormalized).

        Returns
        -------
        tangent : np.ndarray, shape (M, 3)
        """
        cx, cy, cz, _ = self._unpack_params(params)
        dB = _bspline_basis_deriv(t, self._knots_spine, self.degree_spine)  # (M, n_spine)
        dx = dB @ cx
        dy = dB @ cy
        dz = dB @ cz
        return np.stack([dx, dy, dz], axis=-1)  # (M, 3)

    def eval_radius(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the radius function r(t).

        Parameters
        ----------
        t : np.ndarray, shape (M,)
        params : np.ndarray

        Returns
        -------
        r : np.ndarray, shape (M,)
            Radius values (always positive via softplus).
        """
        _, _, _, r_ctrl = self._unpack_params(params)
        B = _bspline_basis(t, self._knots_radius, self.degree_radius)  # (M, n_radius)
        r_raw = B @ r_ctrl
        # Softplus to ensure positivity: log(1 + exp(x))
        return np.log1p(np.exp(r_raw))

    def eval_radius_deriv(self, t: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate dr/dt (derivative of softplus-transformed radius).

        Returns
        -------
        dr : np.ndarray, shape (M,)
        """
        _, _, _, r_ctrl = self._unpack_params(params)
        B = _bspline_basis(t, self._knots_radius, self.degree_radius)
        dB = _bspline_basis_deriv(t, self._knots_radius, self.degree_radius)
        r_raw = B @ r_ctrl
        dr_raw = dB @ r_ctrl
        # Chain rule: d/dt softplus(r_raw(t)) = sigmoid(r_raw) * dr_raw
        sigmoid = 1.0 / (1.0 + np.exp(-r_raw))
        return sigmoid * dr_raw

    # ------------------------------------------------------------------
    # Rotation-minimizing frame
    # ------------------------------------------------------------------

    def compute_frame(
        self,
        t: np.ndarray,
        params: np.ndarray,
        initial_normal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the rotation-minimizing frame {T, N, B} at parameter values t.

        The frame is computed by:
        1. Evaluating the spine and its tangent on a dense grid.
        2. Propagating the Bishop frame along the dense grid.
        3. Interpolating N and B back to the requested t values.

        Parameters
        ----------
        t : np.ndarray, shape (M,)
            Parameter values where the frame is needed.
        params : np.ndarray
            Shape parameters.
        initial_normal : Optional[np.ndarray]
            Initial normal for the Bishop frame propagation.

        Returns
        -------
        T : np.ndarray, shape (M, 3) — unit tangents
        N : np.ndarray, shape (M, 3) — frame normals
        B : np.ndarray, shape (M, 3) — frame binormals
        """
        # Dense grid for stable frame propagation
        t_dense = np.linspace(self.t_low, self.t_high, self.n_frame_samples)

        pos_dense = self.eval_spine(t_dense, params)          # (n_frame, 3)
        tan_dense = self.eval_spine_tangent(t_dense, params)  # (n_frame, 3)

        # Normalize tangents (guard against zero-length segments)
        tan_norms = np.linalg.norm(tan_dense, axis=1, keepdims=True)
        tan_norms = np.maximum(tan_norms, 1e-10)
        tan_dense_unit = tan_dense / tan_norms

        # Propagate Bishop frame on the dense grid
        N_dense, B_dense = _rotation_minimizing_frame(
            pos_dense, tan_dense_unit, initial_normal
        )

        # Interpolate frame back to requested t values using linear interpolation
        # (good enough since the dense grid is fine)
        def interp_vec(vals_dense: np.ndarray, t_query: np.ndarray) -> np.ndarray:
            """Linearly interpolate each column of vals_dense from t_dense to t_query."""
            out = np.empty((len(t_query), 3))
            for d in range(3):
                out[:, d] = np.interp(t_query, t_dense, vals_dense[:, d])
            return out

        T = interp_vec(tan_dense_unit, t)
        N = interp_vec(N_dense, t)
        B_vec = interp_vec(B_dense, t)

        # Re-orthonormalize at query points for numerical cleanliness
        T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-10)
        N = N - (np.sum(N * T, axis=1, keepdims=True)) * T
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-10)
        B_vec = np.cross(T, N)

        return T, N, B_vec

    # ------------------------------------------------------------------
    # Parametrization and Jacobian (wired into Shape)
    # ------------------------------------------------------------------

    def _parametrize(self, t_2d: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Evaluate S(t, theta) = c(t) + r(t) * (cos(theta)*N(t) + sin(theta)*B(t)).

        Parameters
        ----------
        t_2d : np.ndarray, shape (M, 2)
            Each row is (t, theta).
        params : np.ndarray
            Shape parameters.

        Returns
        -------
        points : np.ndarray, shape (M, 3)
        """
        t = t_2d[:, 0]       # spine parameter
        theta = t_2d[:, 1]   # angular parameter

        # Spine position and radius
        c = self.eval_spine(t, params)   # (M, 3)
        r = self.eval_radius(t, params)  # (M,)

        # Frame at these t values
        _, N, B = self.compute_frame(t, params)  # each (M, 3)

        # Surface point
        cos_th = np.cos(theta)[:, np.newaxis]  # (M, 1)
        sin_th = np.sin(theta)[:, np.newaxis]

        return c + r[:, np.newaxis] * (cos_th * N + sin_th * B)

    def _jacobian(self, t_2d: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian ∂S/∂(t, theta), shape (M, 3, 2).

        - Column 0: ∂S/∂t  (partial derivative along spine direction)
        - Column 1: ∂S/∂θ  (partial derivative in angular direction)

        ∂S/∂θ is exact:
            ∂S/∂θ = r(t) * (-sin(θ) * N(t) + cos(θ) * B(t))

        ∂S/∂t uses finite differences (avoids differentiating through the
        frame propagation, which would require differentiating the RMF):
            ∂S/∂t ≈ (S(t+h, θ) - S(t-h, θ)) / (2h)

        Parameters
        ----------
        t_2d : np.ndarray, shape (M, 2)
        params : np.ndarray

        Returns
        -------
        J : np.ndarray, shape (M, 3, 2)
        """
        t = t_2d[:, 0]
        theta = t_2d[:, 1]

        # --- ∂S/∂θ (exact) ---
        r = self.eval_radius(t, params)       # (M,)
        _, N, B = self.compute_frame(t, params)  # (M, 3)
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        dS_dtheta = r[:, np.newaxis] * (
            -sin_th[:, np.newaxis] * N + cos_th[:, np.newaxis] * B
        )  # (M, 3)

        # --- ∂S/∂t (central finite differences) ---
        h = 1e-5 * (self.t_high - self.t_low)
        t_clamp = np.clip(t, self.t_low + h, self.t_high - h)

        t_fwd = np.stack([t_clamp + h, theta], axis=1)
        t_bwd = np.stack([t_clamp - h, theta], axis=1)

        S_fwd = self._parametrize(t_fwd, params)  # (M, 3)
        S_bwd = self._parametrize(t_bwd, params)

        dS_dt = (S_fwd - S_bwd) / (2.0 * h)  # (M, 3)

        # Stack into Jacobian (M, 3, 2)
        J = np.stack([dS_dt, dS_dtheta], axis=-1)
        return J

    # ------------------------------------------------------------------
    # Initialization helper
    # ------------------------------------------------------------------

    @classmethod
    def fit_to_points(
        cls,
        points: np.ndarray,
        n_spine_ctrl: int = 6,
        n_radius_ctrl: int = 4,
        degree_spine: int = 3,
        degree_radius: int = 3,
        spine_axis: Optional[np.ndarray] = None,
    ) -> Tuple["GeneralizedCylinder", np.ndarray]:
        """
        Initialize a GeneralizedCylinder by fitting a straight spine to a point
        cloud and estimating an average radius.

        This provides a good initial guess for further nonlinear optimization.

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)
            Input point cloud.
        n_spine_ctrl : int
            Number of spine control points.
        n_radius_ctrl : int
            Number of radius control points.
        degree_spine, degree_radius : int
            B-spline degrees.
        spine_axis : Optional[np.ndarray], shape (3,)
            Known axis direction. If None, estimated via PCA.

        Returns
        -------
        cyl : GeneralizedCylinder
            The constructed shape.
        params : np.ndarray
            Initial parameter vector.
        """
        # Clamp degrees to be compatible with control point counts
        degree_spine = min(degree_spine, n_spine_ctrl - 1)
        degree_radius = min(degree_radius, n_radius_ctrl - 1)

        cyl = cls(
            n_spine_ctrl=n_spine_ctrl,
            n_radius_ctrl=n_radius_ctrl,
            degree_spine=degree_spine,
            degree_radius=degree_radius,
        )

        # --- NEW: best fitting cylinder ---
        theta_cyl = fit_cylinder_ls(points)

        x0, y0, z0, r0, h0, polar, azim = theta_cyl

        ux = np.sin(polar) * np.cos(azim)
        uy = np.sin(polar) * np.sin(azim)
        uz = np.cos(polar)

        axis = np.array([ux, uy, uz])
        axis /= np.linalg.norm(axis)

        bottom = np.array([x0, y0, z0])
        top = bottom + axis * h0

        t_ctrl = np.linspace(0.0, 1.0, n_spine_ctrl)
        spine_ctrl_pts = bottom + np.outer(t_ctrl * h0, axis)

        # --- compute projection along fitted axis ---
        X = points - bottom
        proj = X @ axis  # scalar coordinate along axis

        # normalize to [0, 1] along cylinder height
        t_norm = np.clip(proj / h0, 0.0, 1.0)

        # --- distance to axis (radius) ---
        proj_clamped = np.clip(proj, 0.0, h0)
        closest = bottom + np.outer(proj_clamped, axis)

        radii = np.linalg.norm(points - closest, axis=1)

        # --- sort for spline fitting ---
        sort_idx = np.argsort(t_norm)
        t_sorted = t_norm[sort_idx]
        r_sorted = radii[sort_idx]

        # Use a simple uniform average per bin as target
        n_bins = n_radius_ctrl * 3
        bin_edges = np.linspace(0, 1, n_bins + 1)
        r_targets = []
        t_targets = []
        for i in range(n_bins):
            mask = (t_sorted >= bin_edges[i]) & (t_sorted < bin_edges[i + 1])
            if mask.sum() > 0:
                t_targets.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                r_targets.append(r_sorted[mask].mean())

        if len(t_targets) < n_radius_ctrl:
            # Fallback: use global mean radius
            r_mean = radii.mean()
            r_ctrl_init = np.full(n_radius_ctrl, r_mean)
        else:
            t_targets = np.array(t_targets)
            r_targets = np.array(r_targets)
            # Fit B-spline to (t_targets, r_targets)
            try:
                spl = make_interp_spline(
                    t_targets, r_targets,
                    k=min(degree_radius, len(t_targets) - 1)
                )
                # Evaluate at the radius control point locations
                t_r_ctrl = np.linspace(0, 1, n_radius_ctrl)
                r_fit = spl(t_r_ctrl)
                # Invert softplus: given r = log(1 + exp(x)), x = log(exp(r) - 1)
                r_fit = np.maximum(r_fit, 1e-3)
                r_ctrl_init = np.log(np.expm1(r_fit))
            except Exception:
                r_ctrl_init = np.full(n_radius_ctrl, radii.mean())

        # Assemble params
        params = np.concatenate([
            spine_ctrl_pts[:, 0],  # cx
            spine_ctrl_pts[:, 1],  # cy
            spine_ctrl_pts[:, 2],  # cz
            r_ctrl_init,
        ])

        return cyl, params

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_spine_params(self, params: np.ndarray) -> np.ndarray:
        """Return spine control points as (n_spine_ctrl, 3) array."""
        cx, cy, cz, _ = self._unpack_params(params)
        return np.stack([cx, cy, cz], axis=1)

    def get_radius_params(self, params: np.ndarray) -> np.ndarray:
        """Return radius control points (before softplus), shape (n_radius_ctrl,)."""
        _, _, _, r_ctrl = self._unpack_params(params)
        return r_ctrl

    def sample_spine(self, params: np.ndarray, n: int = 50) -> np.ndarray:
        """
        Sample the spine curve at n uniformly spaced parameter values.

        Returns
        -------
        pts : np.ndarray, shape (n, 3)
        """
        t = np.linspace(self.t_low, self.t_high, n)
        return self.eval_spine(t, params)

    def smoothness_energy(self, params: np.ndarray) -> float:
        """
        Compute the integrated squared second derivative (bending energy)
        for both the spine and radius, as a regularization term.

            E_smooth = lambda_c * ∫ ||c''(t)||² dt + lambda_r * ∫ r''(t)² dt

        Returns a scalar that can be added to a data-fit objective.
        """
        from scipy.integrate import quad

        cx, cy, cz, r_ctrl = self._unpack_params(params)

        def _second_deriv_energy_1d(ctrl: np.ndarray, knots: np.ndarray, deg: int) -> float:
            """Integrate squared second derivative of a scalar B-spline."""
            spl = BSpline(knots, ctrl, deg)
            d2spl = spl.derivative().derivative()

            def integrand(t_val):
                return float(d2spl(t_val)) ** 2

            val, _ = quad(integrand, self.t_low, self.t_high)
            return val

        # Spine bending: sum over x, y, z components
        E_spine = (
            _second_deriv_energy_1d(cx, self._knots_spine, self.degree_spine)
            + _second_deriv_energy_1d(cy, self._knots_spine, self.degree_spine)
            + _second_deriv_energy_1d(cz, self._knots_spine, self.degree_spine)
        )

        # Radius smoothness
        E_radius = _second_deriv_energy_1d(r_ctrl, self._knots_radius, self.degree_radius)

        return E_spine, E_radius

    def __repr__(self) -> str:
        return (
            f"GeneralizedCylinder("
            f"n_spine_ctrl={self.n_spine_ctrl}, "
            f"n_radius_ctrl={self.n_radius_ctrl}, "
            f"degree_spine={self.degree_spine}, "
            f"degree_radius={self.degree_radius}, "
            f"n_params={self.get_n_params()})"
        )