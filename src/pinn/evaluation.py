# SPDX-License-Identifier: MIT
import torch
from src.utils import print_metrics_table

def evaluate_forward_test_dataset(model, cfd):
    """Evaluate a model on the prepared test dataset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Model to evaluate.
    data : dict
        test prepared dataset mapping.

    Returns
    -------
    None
        Metrics are printed to standard output.
    """

    # CFD test dataset
    data = cfd["test"]

    test_data_available = (
        data["xtest"] is not None 
        and data["ytest"] is not None 
        and data["xtest"].shape[0] > 0
    )

    if not test_data_available:
        print("---------------------------------------")
        print("Skipping test evaluation.")
        print("No test data were created. "
              "This usually means N_test_data = 0 after the "
              "train/validation/test split.")
        return

    test_metrics = model.evaluate_forward(
        data["xtest"], data["ytest"], data["rhotest"], 
        data["utest"], data["vtest"], data["ptest"], 
        data["muttest"]
    )

    # Print metrics of the test dataset
    print_metrics_table(test_metrics, title="Test cfd dataset metrics")

    return test_metrics

def evaluate_inverse_test_dataset(model, observation, cfd=None):
    """Evaluate the available inverse test dataset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Model to evaluate.
    observation : dict or None
        Complete observation dataset mapping, organized by modality.
    cfd : dict or None, optional
        Prepared boundary data with field masks and boundary names.

    Returns
    -------
    dict
        Metrics for available modalities. Boundary metrics are grouped by
        boundary name under ``"boundaries"`` and contain observed fields.
    """

    # Select the test split and prepare nondimensional tensors.
    test_observation = model._prepare_observation_split(
        observation,
        subset="test",
    )

    test_metrics = {}
    if not test_observation:
        print("---------------------------------------")
        print("Skipping test evaluation for observation data.")
        print("No test observation data were created.")
    else:
        with torch.enable_grad():
            test_metrics = model.evaluate_inverse(test_observation)
        print_metrics_table(
            test_metrics, title="Test observation dataset metrics"
        )

    if cfd is not None:
        boundary_metrics = _evaluate_boundary_test_dataset(model, cfd["test"])
        if boundary_metrics:
            test_metrics["boundaries"] = boundary_metrics

    return test_metrics

def _evaluate_boundary_test_dataset(model, data):
    """Evaluate each boundary using only its held-out observed true values."""

    metrics = {}

    if data["xtest"] is None or data["xtest"].shape[0] == 0:
        return {}

    # Corresponding boundary names for each point
    names = data["boundary_names"]

    # Loop over boundary conditions provided
    for name in dict.fromkeys(names):
        selected = names == name
        observed = {
            field: mask[selected]
            for field, mask in data["observed"].items()
        }
        
        metrics[name] = model.evaluate_forward(
            data["xtest"][selected], data["ytest"][selected],
            data["rhotest"][selected], data["utest"][selected],
            data["vtest"][selected], data["ptest"][selected],
            data["muttest"][selected], observed=observed,
        )

        print_metrics_table(
            metrics[name], title=f"Test boundary metrics: {name}"
        )

    return metrics
