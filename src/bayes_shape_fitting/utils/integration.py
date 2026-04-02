import numpy as np
from numpy.polynomial.legendre import leggauss


class Integrator:
    """
    Generic integration engine for testing multiple quadrature rules.
    Supports normal-domain and log-domain integration.

    Methods:
        - trapezoidal
        - simpson
        - romberg (fixed depth)
        - gauss_legendre (N-point)
        - log_trapezoidal (for mixture log-likelihoods)
        - log_simpson
    """

    def __init__(self, method="trapezoidal", gauss_n=32, romberg_depth=5):
        self.method = method
        self.gauss_n = gauss_n
        self.romberg_depth = romberg_depth

    # -------------------------------------------------------------
    # Public API --------------------------------------------------
    # -------------------------------------------------------------

    def integrate(self, t, f):
        """
        Integrate f(t) dt with f in normal domain.
        t: (N,) sorted grid
        f: (M, N) or (N,) values
        """
        if self.method == "trapezoidal":
            return self._trapz(t, f)

        elif self.method == "simpson":
            return self._simpson(t, f)

        elif self.method == "romberg":
            return self._romberg(t, f, depth=self.romberg_depth)

        elif self.method == "gauss":
            return self._gauss_legendre(f_func=f, a=t[0], b=t[-1], n=self.gauss_n)

        else:
            raise ValueError(f"Unknown integration method: {self.method}")

    def log_integrate(self, t, log_f):
        """
        Integrate exp(log_f(t)) dt in log domain using log-sum-exp formulas.
        t: (N,)
        log_f: (M, N)
        """
        if self.method == "trapezoidal":
            return self._log_trapz(t, log_f)

        elif self.method == "simpson":
            return self._log_simpson(t, log_f)

        else:
            raise ValueError(f"log_integrate not implemented for method '{self.method}'")

    # -------------------------------------------------------------
    # Normal-domain integrators -----------------------------------
    # -------------------------------------------------------------

    def _trapz(self, t, f):
        """
        Standard trapezoidal rule. Works with f shape (M,N).
        """
        dt = np.diff(t)
        if f.ndim == 1:
            return np.sum(0.5 * (f[:-1] + f[1:]) * dt)
        return np.sum(0.5 * (f[:, :-1] + f[:, 1:]) * dt[None, :], axis=1)

    def _simpson(self, t, f):
        """
        Composite Simpson rule on nonuniform grids.
        If N is even, last interval uses trapezoid fallback.
        """
        N = len(t)
        if N < 3:
            return self._trapz(t, f)

        if f.ndim == 1:
            f = f[None, :]

        # Simpson works on triples (t[i], t[i+1], t[i+2])
        M = f.shape[0]
        res = np.zeros(M)

        i = 0
        while i + 2 < N:
            t0, t1, t2 = t[i], t[i+1], t[i+2]
            f0, f1, f2 = f[:, i], f[:, i+1], f[:, i+2]

            h0 = t1 - t0
            h1 = t2 - t1

            # One of the generalized Simpson formulas
            denom = h0 + h1
            w0 = -h1**2 / (h0*denom)
            w1 = (h1**2 - h0**2) / (h0*h1)
            w2 = h0**2 / (h1*denom)

            area = (denom / 6.0) * (w0 * f0 + w1 * f1 + w2 * f2)
            res += area

            i += 2

        # fallback trapezoid if odd number of points
        if i + 1 < N:
            res += 0.5 * (f[:, i] + f[:, i+1]) * (t[i+1] - t[i])

        return res

    def _romberg(self, t, f, depth=5):
        """
        Romberg integration on uniform grid only.
        If grid is non-uniform, raises.
        """
        dt = np.diff(t)
        if not np.allclose(dt, dt[0]):
            raise ValueError("Romberg requires uniform grid.")

        if f.ndim == 1:
            f = f[None, :]

        M = f.shape[0]
        N = len(t)
        h = dt[0]

        # R[k, j] approximations: k = level, j = Richardson index.
        R = np.zeros((depth, depth, M))

        # R[0,0]
        R[0, 0] = h * (0.5 * f[:, 0] + np.sum(f[:, 1:-1], axis=1) + 0.5 * f[:, -1])

        for k in range(1, depth):
            h /= 2
            # midpoints contributions
            idx = np.arange(1, 2**k, 2) * (N - 1) // (2**k)
            R[k, 0] = 0.5 * R[k-1, 0] + h * np.sum(f[:, idx], axis=1)

            # Richardson extrapolation
            for j in range(1, k+1):
                R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4**j - 1)

        return R[depth-1, depth-1]

    def _gauss_legendre(self, f_func, a, b, n=32):
        """
        Gauss-Legendre quadrature:
            \int_a^b f(t) dt  ≈ sum w_i f(x_i)
        Here f_func must be callable(t) returning (M,N) or (N,) arrays.
        """
        x, w = leggauss(n)
        t_eval = 0.5*(b - a)*x + 0.5*(b + a)
        f_eval = f_func(t_eval)

        if f_eval.ndim == 1:
            return 0.5*(b - a) * np.sum(w * f_eval)
        return 0.5*(b - a) * np.sum(w[None, :] * f_eval, axis=1)

    # -------------------------------------------------------------
    # Log-domain integrators --------------------------------------
    # -------------------------------------------------------------

    def _log_trapz(self, t, log_f):
        """
        log ∫ f(t) dt  using trapezoidal rule with log-sum-exp.
        log_f: (M, N)
        """
        if log_f.ndim == 1:
            log_f = log_f[None, :]

        dt = np.diff(t)
        M, N = log_f.shape

        # weights for pairs
        log_w1 = np.log(0.5 * dt)
        log_w2 = log_w1

        # log( w1*f[i] + w2*f[i+1] ) = LSE(log_f[:,i] + log w1, log_f[:,i+1] + log w2)
        a = np.maximum(log_f[:, :-1] + log_w1, log_f[:, 1:] + log_w2)
        return np.sum(
            a + np.log(
                np.exp(log_f[:, :-1] + log_w1 - a) +
                np.exp(log_f[:, 1:] + log_w2 - a)
            ),
            axis=1
        )

    def _log_simpson(self, t, log_f):
        """
        log ∫ f(t) dt using Simpson rule with log-sum-exp.
        Generalized 3-point Simpson on non-uniform t.
        """
        if log_f.ndim == 1:
            log_f = log_f[None, :]

        M, N = log_f.shape

        res_terms = []
        i = 0

        while i + 2 < N:
            t0, t1, t2 = t[i], t[i+1], t[i+2]
            h0 = t1 - t0
            h1 = t2 - t1
            denom = h0 + h1

            w0 = (denom / 6) * (-h1**2 / (h0*denom))
            w1 = (denom / 6) * ((h1**2 - h0**2) / (h0*h1))
            w2 = (denom / 6) * (h0**2 / (h1*denom))

            log_w = np.log(np.abs([w0, w1, w2]) + 1e-300)  # avoid log(0)

            logs = log_f[:, i:i+3] + log_w[None, :]        # (M,3)
            a = logs.max(axis=1, keepdims=True)
            term = a[:, 0] + np.log(np.exp(logs - a).sum(axis=1))
            res_terms.append(term)

            i += 2

        # fallback trapezoid
        if i + 1 < N:
            dt = t[i+1] - t[i]
            log_w = np.log(0.5 * dt)
            logs = log_f[:, i:i+2] + log_w
            a = logs.max(axis=1, keepdims=True)
            term = a[:, 0] + np.log(np.exp(logs - a).sum(axis=1))
            res_terms.append(term)

        return np.sum(np.stack(res_terms, axis=1), axis=1)