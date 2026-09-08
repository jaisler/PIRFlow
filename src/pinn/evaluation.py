# SPDX-License-Identifier: MIT
from src.utils import print_metrics_table

def evaluate_test_dataset(model, data):
    """Evaluate a model on the prepared test dataset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Model to evaluate.
    data : dict
        Prepared dataset mapping.

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

    test_metrics = model.evaluate_data(
        data["xtest"], data["ytest"], data["rhotest"], 
        data["utest"], data["vtest"], data["test"]["ptest"], 
        data["muttest"]
    )

    # Print metrics of the test dataset
    print_metrics_table(test_metrics, title="Test dataset metrics")
