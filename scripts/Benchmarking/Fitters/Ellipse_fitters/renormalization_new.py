from ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from utils.ellipse_utils import get_ellipse_params
from scipy.linalg import eig, norm


class RenormalizationFitterNew(BaseEllipseFitter):
    def fit(self, points: np.ndarray, uncertainties:np.ndarray = None,iterations=20, *args, **kwargs) -> np.ndarray:
        """
                Kanatani's renormalization ellipse fitting
                with optional heteroscedastic covariance matrices Σ_i.

                points: Nx2
                uncertainties: None or list/array of N 2×2 covariance matrices
                """

        x = points[:, 0]
        y = points[:, 1]
        Np = len(points)

        # -------------------------------
        # xi(x,y) = [x^2, xy, y^2, x, y, 1]
        # -------------------------------
        def xi_vec(x, y):
            return np.array([x * x, x * y, y * y, x, y, 1.0])

        # Conic gradient w.r.t (x,y)
        def grad_F(x, y, th):
            a, b, c, d, e, _ = th
            Fx = 2 * a * x + b * y + d
            Fy = b * x + 2 * c * y + e
            return np.array([Fx, Fy])

        # Jacobian of xi wrt (x,y)
        def J_xi(x, y):
            return np.array([
                [2 * x, y, 0, 1, 0, 0],
                [0, x, 2 * y, 0, 1, 0]
            ])  # 2×6

        # default covariance = identity
        if uncertainties is None:
            uncertainties = [np.eye(2)] * Np

        # -------------------------------
        # Initial estimate: DLT
        # -------------------------------
        X = np.stack([x * x, x * y, y * y, x, y, np.ones(Np)], axis=1)
        _, _, Vt = np.linalg.svd(X)
        theta = Vt[-1, :]
        theta /= norm(theta)

        # -------------------------------
        # Renormalization iteration
        # -------------------------------
        for it in range(iterations):

            M = np.zeros((6, 6))
            Nmat = np.zeros((6, 6))
            V0_list = []

            # Build M and N
            for i in range(Np):
                xi = xi_vec(x[i], y[i])
                M += np.outer(xi, xi)

                J = J_xi(x[i], y[i])
                Sig = uncertainties[i]
                V0 = J.T @ Sig @ J  # (6×6)
                V0_list.append(V0)

                Nmat += V0

            M /= Np
            Nmat /= Np

            # -------------------------------
            # Solve N θ = λ M θ
            # Renormalization chooses max λ
            # -------------------------------
            eigvals, eigvecs = eig(Nmat, M)
            idx = np.argmax(eigvals.real)
            new_theta = eigvecs[:, idx].real

            # Normalize using M (best practice)
            new_theta /= np.sqrt(new_theta @ M @ new_theta)

            # convergence
            if norm(theta - new_theta) < 1e-12:
                theta = new_theta
                break

            theta = new_theta

        #theta[[1,3,4]] *= 2

        return get_ellipse_params(theta)


