# results.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

import pandas as pd

from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D


@dataclass
class SweepResult:
    experiment_config: "ExperimentConfig"
    fitting_params: np.ndarray  # shape: (num_it, n_fitters, 5)
    fitting_times: np.ndarray
    # ground truth params for reference
    shape_gt: np.ndarray
    source_gt: np.ndarray
    noise_gt: np.ndarray

    def get_fitter_params(self, fitter_label: str) -> np.ndarray:
        """Returns (num_it, 5) for a single fitter, excluding failed iterations."""
        i = self.experiment_config.labels.index(fitter_label)
        params = self.fitting_params[:, i, :]
        return params

    def get_mse(self, fitter_label: str) -> float:
        """MSE against ground truth, averaged over iterations."""
        params = self.get_fitter_params(fitter_label)  # (num_it, 5)
        valid = ~np.isnan(params).any(axis=1)
        return np.mean((params[valid] - self.shape_gt[valid]) ** 2)

    def get_rmse(self, fitter_label: str) -> float:
        """RMSE against ground truth, averaged over iterations."""
        params = self.get_fitter_params(fitter_label)  # (num_it, D)
        valid = ~np.isnan(params).any(axis=1)
        return np.sqrt(np.mean((params[valid] - self.shape_gt[valid]) ** 2))

    def get_fitting_time(self, fitter_label):
        i = self.experiment_config.labels.index(fitter_label)
        return np.mean(self.fitting_times[:, i])

    def save(self, save_dir: str = "results_cache") -> str:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir, self.experiment_config.name)

        np.savez_compressed(
            stem + ".npz",
            fitting_params=self.fitting_params,
            fitting_times=self.fitting_times,
            shape_gt=self.shape_gt,
            source_gt=self.source_gt,
            noise_gt=self.noise_gt,
        )
        return stem

    @classmethod
    def load(cls, stem: str, experiment_config) -> "SweepResult":
        """
        Load a SweepResult from <stem>.npz and <stem>.json.
        experiment_config must be provided externally.
        """
        arrays = np.load(stem + ".npz")


        return cls(
            experiment_config=experiment_config,
            fitting_params=arrays["fitting_params"],
            fitting_times=arrays["fitting_times"],
            shape_gt=arrays["shape_gt"],
            source_gt=arrays["source_gt"],
            noise_gt=arrays["noise_gt"],
        )

    def plot_result(self, save_dir: str = "plots"):

        shape = self.experiment_config.shape
        shape_gt = self.shape_gt[0]
        fitting_params = self.fitting_params
        labels = self.experiment_config.labels
        n_fitters = len(labels)

        cols = math.ceil(n_fitters / 2)
        rows = 2
        plotter = ShapePlotter2D()

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4), sharex=True, sharey=True)
        fig.set_size_inches(19.2, 10.8)
        fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)

        axes = axes.flatten()

        for i, (ax, label) in enumerate(zip(axes, labels)):
            plt.sca(ax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(label, fontsize=20)
            ax.grid(True, alpha=1)

            ax.set_xlabel("")
            ax.set_ylabel("")

            margin_factor_x = 1.5
            margin_factor_y = 2
            x_range = shape_gt[2] * margin_factor_x
            y_range = shape_gt[3] * margin_factor_y
            ax.set_xlim(shape_gt[0] - x_range, shape_gt[0] + x_range)
            ax.set_ylim(shape_gt[1] - y_range, shape_gt[1] + y_range)

            valid = ~np.isnan(fitting_params[:, i, :]).any(axis=1)
            for params_fit in fitting_params[valid, i, :]:
                plotter.plot_shape(shape, params_fit, label="Fitted Ellipses", color="red", ax=ax)

            plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=ax)
            ax.plot(shape_gt[0], shape_gt[1], "ro", color="black", label="True Center")
            ax.plot(
                np.mean(fitting_params[valid, i, 0]),
                np.mean(fitting_params[valid, i, 1]),
                "*", color="blue", label="Averaged Center",
            )

        # in the last spot we add a plot of an example setup
        ax = axes[-1]
        plt.sca(ax)
        # plt.gca().set_aspect("equal", adjustable="box")
        plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="black", ax=axes[-1])
        ax.plot(shape_gt[0], shape_gt[1], "ro", color="black", label="True Center")

        fixed_sigma = self.noise_gt[0]
        # fixed_sigma = self.noise_gt[0][0] # for fixed
        new_noise = copy.copy(self.experiment_config.noise_function)
        new_noise.fixed_params = fixed_sigma

        source_params = self.source_gt[0]
        new_source = copy.copy(self.experiment_config.source_density)
        new_source.fixed_params = source_params

        # compute observations
        parameter_samples = new_source.sample(n=100)
        covariances = new_noise.evaluate(parameter_samples, params=fixed_sigma)
        source_points = shape.parametrization(parameter_samples, shape_gt)
        errors = np.array([np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov) for cov in covariances])
        observations = source_points + errors

        ax.scatter(
            observations[:, 0],
            observations[:, 1], s=8, alpha=0.8, label = "Sampled Points")

        ax.set_xlabel("")
        ax.set_ylabel("")

        # now plot covariances
        # sample_tvals

        for i in range(len(source_points)):
            cov = covariances[i]
            center = source_points[i]

            # Eigen-decomposition of covariance matrix
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            # Compute ellipse angle in degrees
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

            # Width and height = 2 * sqrt(eigenvalues)
            width, height = 2 * np.sqrt(vals)

            ellip_label = label if i == 0 else None

            # Create and add ellipse
            ellip = patches.Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor="green",
                                    facecolor="green", lw=0.1, alpha=0.1, label="Uncertainty")
            plt.gca().add_patch(ellip)

        # plotter.plot_covariances(shape, shape_gt, new_source, new_noise)
        ax.set_title("Example Point Sample", fontsize=20)

        for ax in axes[n_fitters + 1:]:
            ax.set_visible(False)

        # adjust difference between height
        fig.subplots_adjust(hspace=0.01)

        # add legend

        handles, labels = [], []

        for ax in fig.axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

        # Remove duplicates while preserving order
        unique = dict(zip(labels, handles))

        # Add the legend BELOW all subplots, in ONE ROW
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),  # vertical position below figure
            ncol=len(unique),  # <-- one row!
            fontsize=20,
            handletextpad=0.8
        )

        # Add bottom margin so the legend is not cut off
        fig.subplots_adjust(bottom=0.18, top=0.88, hspace=0.35, wspace=0.3)
        plt.tight_layout(rect=[0, 0.08, 1, 0.93])

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.experiment_config.name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot: {save_path}")

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

    def get_all_rmse(self, fitter_label: str) -> np.ndarray:
        """
        RMSE against ground truth, averaged over iterations for each meaningful parameter.

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
            np.sqrt(np.mean(center_dev)),
            np.sqrt(np.mean(a_mse)),
            np.sqrt(np.mean(b_mse)),
            np.sqrt(np.mean(theta_mse)),
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
        i = self.experiment_config.labels.index(fitter_label)
        failed = np.isnan(self.fitting_params[:, i, :]).any(axis=1).sum()
        return failed / len(self.fitting_params)

    def generate_latex_table_Ec_Ea_Earea(self):
        methods = self.experiment_config.labels

        rows = []
        for m in methods:
            Ec = self.get_Ec(m)
            Ea = self.get_Ea(m)
            Earea = self.get_Earea(m)
            F = self.get_failure_rate(m)  # NEW: failure rate
            rows.append([m, Ec, Ea, Earea, F])

        import pandas as pd
        df = pd.DataFrame(rows, columns=["Method", "E_c", "E_a", "E_area", "F_fail"])

        latex = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\renewcommand{\\arraystretch}{1.2}\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|}\n"
        latex += "\\hline\n"
        latex += "Method & $E_c$ & $E_a$ & $E_{\\text{area}}$ & $F_{\\text{fail}}$ \\\\\n"
        latex += "\\hline\n"

        for _, r in df.iterrows():
            latex += (
                f"{r['Method']} & "
                f"{r['E_c']:.4f} & "
                f"{r['E_a']:.4f} & "
                f"{r['E_area']:.4f} & "
                f"{r['F_fail']:.1f}\\% \\\\\n"
            )

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\caption{Offset bias $E_c$, shape deviation $E_a$, area deviation $E_{\\text{area}}$, and failure rate $F_{\\text{fail}}$ for ellipse fitting (averaged over trials).}\n"
        latex += "\\end{table}\n"

        return latex


    def generate_latex_table(self):
        """
        Generate a LaTeX table with vertical column separators and grouped headers.
        """

        methods = self.experiment_config.labels

        # Collect rows
        rows = []
        for m in methods:
            bias = self.get_all_bias(m)
            mse = self.get_all_rmse(m)

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
        latex = "\\begin{table}[h!]\n"
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

    def plot_individual_fits(self, save_dir: str = "plots/individual"):
        """
        Plot a single fit (iteration `idx`) showing ALL fitters in the same plot.

        Shows:
            - Ground truth ellipse
            - All fitted ellipses (one per method)
            - Noisy observations
            - Covariance ellipses
            - Centers (GT + mean fitted per method)
        """

        shape = self.experiment_config.shape
        for idx in range(self.fitting_params.shape[0]):

            shape_gt = self.shape_gt[idx]
            source_params = self.source_gt[idx]
            noise_params = self.noise_gt[idx]

            fitting_params = self.fitting_params
            labels = self.experiment_config.labels
            n_fitters = len(labels)

            plotter = ShapePlotter2D()

            # ------------------------------------------------------------
            #  Create figure
            # ------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"All Fitters — Iteration {idx}", fontsize=22)
            ax.grid(True, alpha=1)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            # ------------------------------------------------------------
            # 1. Plot ground truth ellipse
            # ------------------------------------------------------------
            plotter.plot_shape(shape, shape_gt, color="black", ax=ax, linewidth=2, label="Ground Truth")
            ax.plot(shape_gt[0], shape_gt[1], "o", color="black")

            # ------------------------------------------------------------
            # 2. Plot each fitted ellipse
            # ------------------------------------------------------------
            color_cycle = plt.cm.tab10(np.linspace(0, 1, n_fitters))

            for i, label in enumerate(labels):
                params_fit = fitting_params[idx, i]

                if np.isnan(params_fit).any():
                    continue  # skip failed fit

                plotter.plot_shape(shape, params_fit,
                                   color=color_cycle[i],
                                   ax=ax,
                                   linewidth=2.5,
                                   label=label)

            # ------------------------------------------------------------
            # 3. Generate noisy observations (same as in plot_result)
            # ------------------------------------------------------------
            # source
            new_source = copy.copy(self.experiment_config.source_density)
            new_source.fixed_params = source_params

            # noise
            new_noise = copy.copy(self.experiment_config.noise_function)
            new_noise.fixed_sigma = noise_params[0]

            # sample measurement
            parameter_samples = new_source.sample(n=100)
            covariances = new_noise.evaluate(parameter_samples, params=np.array([noise_params[0]]))
            source_points = shape.parametrization(parameter_samples, shape_gt)

            errors = np.array([
                np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov)
                for cov in covariances
            ])
            observations = source_points + errors

            # plot points
            ax.scatter(observations[:, 0], observations[:, 1], s=10, alpha=0.3, color="gray")

            # ------------------------------------------------------------
            # 4. Plot covariance ellipses
            # ------------------------------------------------------------
            # plotter.plot_covariances(shape, shape_gt, new_source, new_noise)

            # ------------------------------------------------------------
            # 5. Auto scale
            # ------------------------------------------------------------
            margin_factor_x = 1.2
            margin_factor_y = 3
            x_range = shape_gt[2] * margin_factor_x
            y_range = shape_gt[3] * margin_factor_y
            ax.set_xlim(shape_gt[0] - x_range, shape_gt[0] + x_range)
            ax.set_ylim(shape_gt[1] - y_range, shape_gt[1] + y_range)

            ax.legend(loc="upper right", fontsize=10)

            # ------------------------------------------------------------
            # 6. Save
            # ------------------------------------------------------------
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.experiment_config.name}_all_fit_{idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"Saved combined fit plot: {save_path}")

    def plot_cylinder(self):
        shape = self.experiment_config.shape
        shape_gt = self.shape_gt[0]
        fitting_params = self.fitting_params
        labels = self.experiment_config.labels
        n_fitters = len(labels)
        color_cycle = plt.cm.tab10(np.linspace(0, 1, n_fitters))

        plotter = ShapePlotter3D()

        # plot the ground_truth first
        fig, ax, = plotter.plot_shape(shape, shape_gt, label="Ground Truth", color="red")

        fixed_sigma = self.noise_gt[0]
        new_noise = copy.copy(self.experiment_config.noise_function)
        new_noise.fixed_params = fixed_sigma

        source_params = self.source_gt[0]
        new_source = copy.copy(self.experiment_config.source_density)
        new_source.fixed_params = source_params

        # compute observations
        parameter_samples = new_source.sample(n=100)
        covariances = new_noise.evaluate(parameter_samples, params=fixed_sigma)
        source_points = shape.parametrization(parameter_samples, shape_gt)
        errors = np.array([np.random.multivariate_normal(np.zeros(shape.ambient_dim), cov) for cov in covariances])
        observations = source_points + errors

        ax.scatter3D(
            observations[:, 0],
            observations[:, 1], observations[:,2], s=8, alpha=0.8)

        for i, label in enumerate(labels):
            plotter.plot_shape(shape, fitting_params[0,i,:], ax=ax, color=color_cycle[i], label=label)

        plt.legend()
        plt.show()

    def get_cylinder_param_errors(self, fitter_label: str) -> dict:
        """
        Compute per-parameter squared error, MSE, and RMSE for the 7 cylinder
        parameters:
            [x_c, y_c, z_c, r, h, polar, azim]

        Handles angular periodicity for polar and azim.

        Returns a dictionary:
            {
                "SE": {param: array of squared errors per iteration},
                "MSE": {param: float},
                "RMSE": {param: float},
                "valid_mask": [...]
            }
        """
        # parameter names in order
        names = ["x_c", "y_c", "z_c", "r", "h", "polar", "azim"]

        # Extract estimated parameters from this fitter
        P = self.get_fitter_params(fitter_label)  # shape (num_it, 7)
        GT = self.shape_gt[:, :7]  # ensure same shape

        # Mask invalid runs
        valid = ~np.isnan(P).any(axis=1)
        P = P[valid]
        GT = GT[valid]

        if len(P) == 0:
            raise ValueError(f"No valid runs for fitter '{fitter_label}'")

        # Compute raw differences
        diffs = P - GT

        # Handle angle periodicity: bring diffs to (-pi, pi]
        diffs[:, -2] = (diffs[:, -2] + np.pi) % (2 * np.pi) - np.pi  # polar
        diffs[:, -1] = (diffs[:, -1] + np.pi) % (2 * np.pi) - np.pi  # azim

        # Squared error per iteration
        SE = diffs ** 2

        # MSE and RMSE per parameter
        MSE = SE.mean(axis=0)
        RMSE = np.sqrt(MSE)

        return {
            "SE": dict(zip(names, [SE[:, i] for i in range(7)])),
            "MSE": dict(zip(names, MSE)),
            "RMSE": dict(zip(names, RMSE)),
            "valid_mask": valid,
        }

    def print_cylinder_comparison_table(self):
        """
        Print a single table comparing ALL fitters across all 7 cylinder
        parameters:
            [x_c, y_c, z_c, r, h, polar, azim]

        Shows MSE for each parameter.
        """

        param_names = ["x_c", "y_c", "z_c", "r", "h", "polar", "azim"]

        # Header
        print("\n================ Cylinder Parameter MSE Comparison ================\n")
        header = ["Fitter"] + param_names
        print("{:<15}".format(header[0]) + " ".join(f"{p:>12}" for p in header[1:]))
        print("-" * (15 + 12 * len(param_names)))

        # One row per fitter
        for fitter in self.experiment_config.labels:
            errs = self.get_cylinder_param_errors(fitter)
            row = [fitter] + [errs["MSE"][p] for p in param_names]

            print("{:<15}".format(row[0]) + " ".join(f"{v:12.6f}" for v in row[1:]))

        print("-" * (15 + 12 * len(param_names)))
        print()
