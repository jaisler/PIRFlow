# SPDX-License-Identifier: MIT
"""Coordinate configured dataset preparation, model training, and plotting."""

from src.config import create_output_directories, load_config
from src.networks import build_network
from src.observation import ObservationData, prepare_observation_datasets
from src.postprocessing import run_flowfield_postprocessing
from src.pinn import (
    build_pinn_model, 
    evaluate_test_dataset, 
    train_model,
)
from src.sampling import (
    get_collocation_points,
    get_data_points,
    prepare_cfd_datasets,
    prepare_collocation_dataset,
)
from src.utils import (
    plot_observation_data,
    plot_prepared_observation_datasets,
    plot_prepared_sampling_datasets,
    plot_sampling_data,
    plot_schlieren_image,
)


def run() -> None:
    """Execute the configured PIRFlow reconstruction workflow.

    Load the configuration, prepare CFD and collocation datasets, and plot
    the sampled points and dataset splits. For inverse problems, load,
    prepare, and plot observations, then return before model training.
    For forward problems, build and train the model, evaluate the test
    dataset, and run flow-field postprocessing when enabled.

    Returns
    -------
    None
        Results are written to the configured output directories.
    """

    # Configuration file
    params = load_config()

    # Create output directories
    create_output_directories(params)

    # Sample or read collocation points if pinn model
    # Required for both forward and inverse problems.
    collocation_pnts = get_collocation_points(params)

    # Sample or read data points if forward problem
    data_pnts = get_data_points(params)

    # Plot original sampling points
    plot_sampling_data(data_pnts, collocation_pnts, params)

    # Prepare training, validation, test and collocation datasets
    cfd_datasets = prepare_cfd_datasets(data_pnts, params)

    # Collocation points must be used the inverse problem as well
    collocation_dataset = prepare_collocation_dataset(
        collocation_pnts, params) 

    # Plot prepared datasets from sampling
    plot_prepared_sampling_datasets(cfd_datasets, collocation_dataset, params)

    # Build neural network
    network = build_network(params)

    # Problem definition
    problem = params["run"].get("problem", "forward").lower()

    # Load observation data
    if problem == "inverse":
        # Load and organize observation data for the inverse problem
        observation_loader = ObservationData(params)

        # Raw observation dataset
        observations = observation_loader.load_observation_data()

        # Split observation dataset
        observation_datasets = prepare_observation_datasets(
            observations,
            params,
        )

        # Plot all observation points
        plot_observation_data(observations, params)

        # Plot Schlieren image
        if params["identification"]["observations"]["schlieren"].get("enabled", False):
            plot_schlieren_image(observations["schlieren"], params)

        # Plot observation datasets: training, validation, test
        plot_prepared_observation_datasets(observation_datasets, params)

        return

    else:
        observation_datasets = None

    # Build model
    # The model is built with the datasets, but for the inverse problem,
    # we need to pass the observation data instead of datasets.
    model = build_pinn_model(
        network, 
        params,
        cfd_datasets,
        observation_datasets, 
        collocation_dataset, 
    )

    # Train model
    train_model(model, params)

    # Evaluate test dataset
    evaluate_test_dataset(model, cfd_datasets["test"])

    # Postprocess flowfield
    if params["run"]["routines"].get("postprocessing", False):
        run_flowfield_postprocessing(model, params)
