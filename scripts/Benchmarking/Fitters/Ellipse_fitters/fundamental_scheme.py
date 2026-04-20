import numpy as np
from ellipse_fitters.base import BaseEllipseFitter
from ellipse_fitters.least_squares import LeastSquaresFitter
from utils.ellipse_utils import get_ellipse_params, get_homogenous_coords_from_ellipse

from utils.ellipse_utils import get_ellipse_params
from utils.plot_utils import plot_ellipse
import copy

"""class FundamentalSchemeFitter(BaseEllipseFitter):

    def fit(self, points, uncertainties: np.ndarray = None, iterations: int = 1, *args, **kwargs):

    

        # first, we compute the initial fit with least squares
        L = LeastSquaresFitter()
        alg_fit = L.fit(points)
        # convert it to homogenous coordinates
        theta_zero = get_homogenous_coords_from_ellipse(alg_fit)

        u = lambda x,y: np.array([[x ** 2, x * y, y ** 2, x, y, 1]])
        grad_u = lambda x, y: np.array([[2 * x, y, 0, 1, 0, 0], [0, x, 2 * y, 0, 1, 0]])

        x = points[:, 0]
        y = points[:, 1]

        j = 0
        diff = 10000.0

        while diff > 1e-10 and j < iterations:
            A_sum = np.zeros((6, 6))
            B_sum = np.zeros((6, 6))
            for i in range(len(points)):
                x_i = x[i]
                y_i = y[i]
                A_i = u(x_i,y_i).T @ u(x_i,y_i)
                B_i = grad_u(x_i,y_i).T @ uncertainties[i] @ grad_u(x_i,y_i)
                denom = max(theta_zero.T @ B_i @ theta_zero, 1e-5)
                # update the sums
                A_sum += A_i / denom
                B_sum += (theta_zero.T @ A_i @ theta_zero) / (denom **2) * B_i

            M = A_sum
            L = B_sum
            X = (M - L + (M - L).T) / 2  # ensure symmetry

            eig_val, eig_vec = np.linalg.eigh(X)
            idx = np.argmin(np.abs(eig_val))
            theta = eig_vec[:, idx]

            diff = np.linalg.norm(theta_zero - theta)
            theta /= np.linalg.norm(theta)
            theta_zero = theta
            j += 1

        print(j)
        return get_ellipse_params(theta)
"""


class FundamentalSchemeFitter(BaseEllipseFitter):

    def fit(self, points, uncertainties: np.ndarray = None, iterations: int = 100, *args, **kwargs):


        noise_x = points[:, 0]
        noise_y = points[:, 1]
        f = 1

        W = np.ones(len(noise_x), dtype="float64")

        # first, we compute the initial fit with least squares
        L = LeastSquaresFitter()
        alg_fit = L.fit(points)
        # convert it to homogenous coordinates
        theta_zero = get_homogenous_coords_from_ellipse(alg_fit)
        theta_zero[[1, 3, 4]] /= 2

        diff = 10000.0

        j = 0

        grad_u = lambda x, y: np.array([[2 * x, 2 * y, 0, 2, 0, 0], [0, 2 * x, 2 * y, 0, 2, 0]])

        while diff > 1e-10 and j < iterations:
            xi_sum = np.zeros((6, 6))
            L_sum = np.zeros((6, 6))
            V0_xi_list = []
            for i in range(len(noise_x)):
                x = noise_x[i]
                y = noise_y[i]
                xi = np.array([[x ** 2, 2 * x * y, y ** 2, 2 * f * x, 2 * f * y, f * f]])
                xi_sum += np.dot(W[i], np.dot(xi.T, xi))

                if uncertainties is None:
                    V0_xi = 4 * np.array(
                        [
                            [x ** 2, x * y, 0, f * x, 0, 0],
                            [x * y, x ** 2 + y ** 2, x * y, f * y, f * x, 0],
                            [0, x * y, y ** 2, 0, f * y, 0],
                            [f * x, f * y, 0, f ** 2, 0, 0],
                            [0, f * x, f * y, 0, f ** 2, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    )
                else:
                    V0_xi = grad_u(x, y).T @ uncertainties[i] @ grad_u(x, y)
                V0_xi_list.append(V0_xi)
                L_sum += np.dot(
                    np.dot(W[i] ** 2, np.dot(xi.T[:, 0], theta_zero) ** 2), V0_xi
                )


            M = xi_sum / len(noise_x)
            L = L_sum / len(noise_x)
            X = M - L
            eig_val, eig_vec = np.linalg.eig(X)
            theta = eig_vec[:, np.argmin(np.abs(eig_val))]

            if np.dot(theta, theta_zero) < 0:
                theta = np.dot(-1, theta)
            diff = np.sum(np.abs(theta_zero - theta))
            if diff <= 1e-10:
                break

            theta /= np.linalg.norm(theta)

            # Compute all weights at once
            W = np.array([1.0 / (theta.T @ V0_xi_list[i] @ theta + 1e-6) for i in range(len(noise_x))])
            W /= np.mean(W)

            theta_zero = theta
            j = j + 1
            if j == iterations - 1:
                print("No convergence")
                return [np.nan]*5
           
        theta[[1, 3, 4]] *= 2

        return get_ellipse_params(theta)
