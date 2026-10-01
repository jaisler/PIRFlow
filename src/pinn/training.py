# SPDX-License-Identifier: MIT
import time

from src.utils import plot_history_training

def training_is_enabled(params):
    """Check whether an optimizer has a positive training budget.

    Parameters
    ----------
    params : dict
        PIRFlow configuration.

    Returns
    -------
    bool
        Whether Adam or L-BFGS is enabled with positive iterations.
    """

    adam_enabled = params["optimizer"]["adam"].get("enabled", False)
    lbfgs_enabled = params["optimizer"]["lbfgs"].get("enabled", False)

    adam_iterations = int(
        params["optimizer"]["adam"].get("iterations", 0)
    )

    lbfgs_iterations = int(
        params["optimizer"]["lbfgs"].get("iterations", 0)
    )

    return (
        adam_enabled and adam_iterations > 0
    ) or (
        lbfgs_enabled and lbfgs_iterations > 0
    )

def train_model(model, params):
    """Train a model and optionally save its checkpoint.

    Parameters
    ----------
    model : PhysicsInformedNN
        Model to train.
    params : dict
        PIRFlow configuration.

    Returns
    -------
    None
        Training state and output files are updated in place.
    """

    # Check if training is enabled
    do_training = training_is_enabled(params)

    # Train
    if not do_training:
        print("---------------------------------------")
        print("Skipping training. Adam and LBFGS are disabled or both have "
              "zero iterations.")
        return

    start_time = time.time()                
    model.fit()
    elapsed = time.time() - start_time
    print("---------------------------------------")                
    print('Training time: %.4f' % (elapsed))

    # Save model
    if params['run']['checkpoint']['save_model']:
        model.save_model(
            params['paths']['model'], 
            params['files']['model_name']
        )

    # Plot history training
    plot_history_training(model, params)
