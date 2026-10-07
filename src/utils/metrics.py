# SPDX-License-Identifier: MIT
import torch

@torch.no_grad()
def compute_metrics(y_pred, y_true, eps=1.0e-12):
    """Compute metrics for one scalar variable.

    Parameters
    ----------
    y_pred, y_true : torch.Tensor
        Matching, nonempty tensors containing one scalar variable.
    eps : float
        Absolute cutoff for reference RMS and standard deviation,
        expressed in the same units as the evaluated values.

    Returns
    -------
    dict
        MSE, RMSE, relative L2, and R2.
        Unavailable relative L2 or R2 values are returned as NaN.
    """

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Prediction shape {tuple(y_pred.shape)} does not match "
            f"reference shape {tuple(y_true.shape)}."
        )

    if y_true.numel() == 0:
        raise ValueError("Cannot compute metrics on empty tensors.")

    # Use double precision for metric arithmetic.
    pred = y_pred.detach().to(dtype=torch.float64)
    true = y_true.detach().to(dtype=torch.float64)

    if not (
        torch.isfinite(pred).all().item()
        and torch.isfinite(true).all().item()
    ):
        raise ValueError(
            "Predictions and references must contain finite values."
        )

    error = pred - true
    mse = torch.mean(error.square())
    rmse = torch.sqrt(mse)

    ref_rms = torch.sqrt(torch.mean(true.square()))

    ref_variance = torch.mean(
        (true - torch.mean(true)).square()
    )
    ref_std = torch.sqrt(ref_variance)

    # Relative L2 needs a nonzero reference magnitude.
    rel_l2 = float("nan")
    if ref_rms.item() > eps:
        rel_l2 = (rmse / ref_rms).item()

    # R2 needs at least two samples and reference variation.
    r2 = float("nan")
    if true.numel() >= 2 and ref_std.item() > eps:
        r2 = (1.0 - mse / ref_variance).item()

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "rel_l2": rel_l2,
        "r2": r2,
    }

def print_metrics_table(metrics, title="Metrics", rel_l2_percent=True):
    """Print a formatted metric table for each flow variable.

    Parameters
    ----------
    metrics : dict
        Metrics keyed by flow variable.
    title : str, optional
        Table title.
    rel_l2_percent : bool, optional
        Whether to display relative L2 error as a percentage.

    Returns
    -------
    None
        The table is written to standard output.
    """

    rel_l2_name = "Rel. L2 (%)" if rel_l2_percent else "Rel. L2"

    header = (
        f"{'Variable':<10}"
        f"{'MSE':>17}"
        f"{'RMSE':>17}"
        f"{rel_l2_name:>17}"
        f"{'R2':>17}"
    )

    line_width = len(header)

    print("=" * line_width)
    print(f"{title:}")
    print("-" * line_width)
    print(header)
    print("-" * line_width)

    preferred_order = ["rho", "u", "v", "p", "mut",
                       "sch_obs", "u_obs", "v_obs", "p_obs"]

    for var in preferred_order:
        if var not in metrics:
            continue

        values = metrics[var]

        mse = values["mse"]
        rmse = values["rmse"]
        rel_l2 = values["rel_l2"]
        r2 = values["r2"]

        if rel_l2_percent:
            rel_l2 = 100.0 * rel_l2
        
        print(
            f"{var:<10} | "
            f"{mse:>14.4e} | "
            f"{rmse:>14.4e} | "
            f"{rel_l2:>14.4e} | "
            f"{r2:>14.4e}"
        )

    print("=" * line_width)
