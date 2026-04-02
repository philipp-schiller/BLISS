# results.py
import numpy as np
from dataclasses import dataclass
from scripts.Benchmarking.ground_truth_config import GroundTruthConfig
import pandas as pd


@dataclass
class SweepResult:
    case_name: str
    noise_name: str
    source_name: str
    ground_truth_config: GroundTruthConfig
    fitter_labels: list[str]
    fitting_params: np.ndarray # shape: (num_it, n_fitters, 5)
    fitting_times: np.ndarray
    # ground truth params for reference
    shape_gt: np.ndarray
    source_gt: np.ndarray
    noise_gt: np.ndarray

    def get_fitter_params(self, fitter_label: str) -> np.ndarray:
        """Returns (num_it, 5) for a single fitter, excluding failed iterations."""
        i = self.fitter_labels.index(fitter_label)
        params = self.fitting_params[:, i, :]
        #return params[~np.isnan(params).any(axis=1)]
        return params


    def get_mse(self, fitter_label: str) -> float:
        """MSE against ground truth, averaged over iterations."""
        params = self.get_fitter_params(fitter_label)  # (num_it, 5)
        valid = ~np.isnan(params).any(axis=1)
        return np.mean((params[valid] - self.shape_gt[valid]) ** 2)

    def get_fitting_time(self, fitter_label):
        i = self.fitter_labels.index(fitter_label)
        return np.mean(self.fitting_times[:,i])

    def get_Ec(self, fitter_label: str) -> float:
        """
        Offset bias:
            Ec = ||c_t - c_hat||_2   averaged over iterations
        """
        params = self.get_fitter_params(fitter_label)  # (num_it, 5)
        valid = ~np.isnan(params).any(axis=1)
        P = params[valid]
        GT = self.shape_gt[valid]

        diff = P[:, :2] - GT[:, :2]  # (cx, cy) difference
        Ec = np.linalg.norm(diff, axis=1)  # Euclidean distance
        return float(np.mean(Ec))

    def get_Ea(self, fitter_label: str) -> float:
        """
        Shape deviation according to:
            Ea = Smax(A_hat^{-1} A_t) / Smin(A_hat^{-1} A_t) - 1

        Averaged across valid iterations.

        """

        def ellipse_shape_matrix(params):
            """
            Build the 2×2 positive semi-definite shape matrix A
            for an ellipse parameterized by (cx, cy, a, b, phi).
            """
            cx, cy, a, b, phi = params
            R = np.array([
                [np.cos(phi), -np.sin(phi)],
                [np.sin(phi), np.cos(phi)],
            ])
            return R @ np.diag([a, b]) @ R.T

        params = self.get_fitter_params(fitter_label)
        valid = ~np.isnan(params).any(axis=1)

        P = params[valid]
        GT = self.shape_gt[valid]

        Ea_values = []

        for p, gt in zip(P, GT):
            A_hat = ellipse_shape_matrix(p)
            A_t = ellipse_shape_matrix(gt)

            # residual transformation
            M = np.linalg.inv(A_hat) @ A_t

            # singular values
            svals = np.linalg.svd(M, compute_uv=False)
            Ea = svals.max() / svals.min() - 1.0
            Ea_values.append(Ea)

        return float(np.mean(Ea_values))

    def get_Earea(self, fitter_label: str) -> float:
        """
        Relative ellipse area deviation:
            E_area = | (a_hat * b_hat) / (a_true * b_true) - 1 |
        averaged over iterations.
        """
        params = self.get_fitter_params(fitter_label)
        valid = ~np.isnan(params).any(axis=1)

        P = params[valid]
        GT = self.shape_gt[valid]

        a_hat = P[:, 2]
        b_hat = P[:, 3]
        a_t = GT[:, 2]
        b_t = GT[:, 3]

        area_ratio = (a_hat * b_hat) / (a_t * b_t)
        Earea = np.abs(area_ratio - 1.0)

        return float(np.mean(Earea))

    def get_all_mse(self, fitter_label: str) -> np.ndarray:
        """
        MSE against ground truth, averaged over iterations for each meaningful parameter.

        Output order:
            [ center deviation, a-error, b-error, theta-error ]
        """
        params = self.get_fitter_params(fitter_label)  # (num_it, 5)
        valid = ~np.isnan(params).any(axis=1)

        # select valid
        P = params[valid]
        GT = self.shape_gt[valid]

        # --- compute deviations ---
        # center deviation as *distance*, not per-component MSE
        center_dev = (P[:, 0] - GT[:, 0]) ** 2 + (P[:, 1] - GT[:, 1]) ** 2
        center_dev = np.sqrt(center_dev)  # shape (num_valid,)

        # a, b, theta MSE (per parameter)
        a_mse = (P[:, 2] - GT[:, 2]) ** 2
        b_mse = (P[:, 3] - GT[:, 3]) ** 2
        theta_mse = (P[:, 4] - GT[:, 4]) ** 2

        # average over iterations
        return np.array([
            np.mean(center_dev),
            np.mean(a_mse),
            np.mean(b_mse),
            np.mean(theta_mse),
        ])

    def get_all_bias(self, fitter_label: str) -> np.ndarray:
        """
        Mean signed error (bias) for:
        [center deviation, a, b, theta]
        """
        params = self.get_fitter_params(fitter_label)
        valid = ~np.isnan(params).any(axis=1)

        P = params[valid]
        GT = self.shape_gt[valid]

        # center bias as signed *radial* deviation
        cx_err = P[:, 0] - GT[:, 0]
        cy_err = P[:, 1] - GT[:, 1]
        center_dev = np.sqrt(cx_err ** 2 + cy_err ** 2)  # radial distance

        a_bias = P[:, 2] - GT[:, 2]
        b_bias = P[:, 3] - GT[:, 3]
        theta_bias = P[:, 4] - GT[:, 4]

        return np.array([
            np.mean(center_dev),
            np.mean(a_bias),
            np.mean(b_bias),
            np.mean(theta_bias),
        ])

    def get_all_std(self, fitter_label: str) -> np.ndarray:
        """
        Standard deviation of errors for:
        [center deviation, a, b, theta]
        """
        params = self.get_fitter_params(fitter_label)
        valid = ~np.isnan(params).any(axis=1)

        P = params[valid]
        GT = self.shape_gt[valid]

        # Center deviation errors
        cx_err = P[:, 0] - GT[:, 0]
        cy_err = P[:, 1] - GT[:, 1]
        center_dev = np.sqrt(cx_err ** 2 + cy_err ** 2)

        a_err = P[:, 2] - GT[:, 2]
        b_err = P[:, 3] - GT[:, 3]
        theta_err = P[:, 4] - GT[:, 4]

        return np.array([
            np.std(center_dev),
            np.std(a_err),
            np.std(b_err),
            np.std(theta_err),
        ])

    def get_failure_rate(self, fitter_label: str) -> float:
        i = self.fitter_labels.index(fitter_label)
        failed = np.isnan(self.fitting_params[:, i, :]).any(axis=1).sum()
        return failed / len(self.fitting_params)

    def generate_latex_table_Ec_Ea_Earea(self):
        methods = self.fitter_labels

        rows = []
        for m in methods:
            Ec = self.get_Ec(m)
            Ea = self.get_Ea(m)
            Earea = self.get_Earea(m)
            rows.append([m, Ec, Ea, Earea])

        import pandas as pd
        df = pd.DataFrame(rows, columns=["Method", "E_c", "E_a", "E_area"])

        latex = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex += "\\begin{tabular}{|l|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Method & $E_c$ & $E_a$ & $E_{\\text{area}}$ \\\\\n"
        latex += "\\hline\n"

        for _, r in df.iterrows():
            latex += f"{r['Method']} & {r['E_c']:.4f} & {r['E_a']:.4f} & {r['E_area']:.4f} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Offset bias $E_c$, shape deviation $E_a$, and area deviation $E_{\\text{area}}$ for ellipse fitting (averaged over trials).}\n"
        latex += "\\end{table}\n"

        return latex

    def generate_latex_table(self):
        """
        Generate a LaTeX table with vertical column separators and grouped headers.
        """
    
        methods = self.fitter_labels
    
        # Collect rows
        rows = []
        for m in methods:
            bias = self.get_all_bias(m)
            mse  = self.get_all_mse(m)
    
            row = [m] + \
                  [bias[0], mse[0]] + \
                  [bias[1], mse[1]] + \
                  [bias[2], mse[2]] + \
                  [bias[3], mse[3]]
    
            rows.append(row)
    
        # DataFrame (not required, but convenient)
        cols = [
            "Method",
            "Center Bias", "Center MSE",
            "a Bias", "a MSE",
            "b Bias", "b MSE",
            "theta Bias", "theta MSE",
        ]
        df = pd.DataFrame(rows, columns=cols)
    
        # -----------------------------
        # LaTeX TABLE WITH COLUMN LINES
        # -----------------------------
        latex  = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex += "\\begin{tabular}{|l||cc||cc||cc||cc|}\n"
        latex += "\\hline\n"
    
        latex += (
            "\\multirow{2}{*}{Method} & "
            "\\multicolumn{2}{c||}{Center} & "
            "\\multicolumn{2}{c||}{a} & "
            "\\multicolumn{2}{c||}{b} & "
            "\\multicolumn{2}{c|}{$\\theta$} \\\\\n"
        )
    
        latex += "\\cline{2-9}\n"
        latex += " & Bias & MSE & Bias & MSE & Bias & MSE & Bias & MSE \\\\\n"
        latex += "\\hline\n"
    
        # Add rows
        for _, r in df.iterrows():
            latex += f"{r['Method']} & "
            latex += f"{r['Center Bias']:.4f} & {r['Center MSE']:.4f} & "
            latex += f"{r['a Bias']:.4f} & {r['a MSE']:.4f} & "
            latex += f"{r['b Bias']:.4f} & {r['b MSE']:.4f} & "
            latex += f"{r['theta Bias']:.4f} & {r['theta MSE']:.4f} \\\\\n"
    
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Bias and MSE for ellipse parameter estimates over 50 runs.}\n"
        latex += "\\end{table}\n"
    
        return latex
