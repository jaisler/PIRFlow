# SPDX-License-Identifier: MIT
import torch
from src.utils import print_metrics_table


def evaluate_test_cfd_dataset(model, data):
    """Evaluate a model on the prepared test subset.
    
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

    test_metrics = model._evaluate_cfd(
        data["xtest"], data["ytest"], data["rhotest"], 
        data["utest"], data["vtest"], data["ptest"], 
        data["muttest"]
    )

    # Print metrics of the test dataset
    print_metrics_table(test_metrics, title="Test cfd dataset metrics")

    return test_metrics

def evaluate_test_observation_dataset(model, observation):
    """Evaluate the available test observation subset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Model to evaluate.
    observation : dict or None
        Complete observation dataset mapping, organized by modality.

    Returns
    -------
    dict
        Metrics for the available test modalities, or an empty dictionary
        when no test observations are available.
    """

    # Select the test split and prepare nondimensional tensors.
    test_observation = model._prepare_observation_split(
        observation,
        subset="test",
    )

    if not test_observation:
        print("---------------------------------------")
        print("Skipping test evaluation for observation data.")
        print("No test observation data were created.")
        return {}

    with torch.enable_grad():
        test_metrics = model._evaluate_observation(test_observation)

    # Print metrics of the test dataset
    print_metrics_table(test_metrics, title="Test observation dataset metrics")

    return test_metrics