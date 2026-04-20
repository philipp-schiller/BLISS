from ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from scripts.Benchmarking.Fitters.Ellipse_fitters.ellipse_utils import get_ellipse_params
from scipy.linalg import eig


class RenormalizationFitter(BaseEllipseFitter):
    def fit(self, points: np.ndarray, uncertainties:np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Fits an ellipse to data via renormalization
        :param uncertainties:
        :param points:
        :param kwargs:
        :return:
        """

        noise_x = points[:, 0]
        noise_y = points[:, 1]
        f = 1

        W = np.ones(len(noise_x), dtype="float64")
        theta_zero = np.zeros(6, dtype="float64")
        diff = 10000.0
        j = 0

        grad_u = lambda x, y: np.array([[2 * x, 2 * y, 0, 2, 0, 0], [0, 2 * x, 2 * y, 0, 2, 0]])

        while diff > 1e-10 and j < 20:
            xi_sum = np.zeros((6, 6))
            N_sum = np.zeros((6, 6))
            V0_xi_list = []
            for i in range(len(noise_x)):
                x = noise_x[i]
                y = noise_y[i]
                xi = np.array([[x ** 2, 2 * x * y, y ** 2, 2 * f * x, 2 * f * y, f * f]])
                xi_sum += np.dot(np.dot(W[i], xi.T), xi)
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
                N_sum += np.dot(W[i], V0_xi)

            M = xi_sum / len(noise_x)
            N = N_sum / len(noise_x)
            eig_val, eig_vec = eig(N, M)
            theta = eig_vec[:, np.argmax(eig_val)]

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
            j = j+1

        theta[[1,3,4]] *= 2

        return get_ellipse_params(theta)


