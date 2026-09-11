# SPDX-License-Identifier: MIT
"""Build physics-informed model wrappers from prepared datasets."""

from .physics_informed_nn import PhysicsInformedNN

def build_pinn_model(
        network, 
        params,
        *, 
        cfd_datasets=None, 
        observation_datasets=None, 
        collocation_dataset=None
):
    """Build a configured physics-informed model wrapper.

    Parameters
    ----------
    network : torch.nn.Module
        MLP or GNN used for field prediction.
    params : dict
        PIRFlow configuration.
    cfd_datasets : dict or None, optional
        Prepared CFD datasets containing ``"training"`` and ``"validation"``
        mappings with suffixed coordinate and flow-field keys, such as
        ``"xtrain"`` and ``"rhoval"``. Training data are required for a
        forward problem.
    observation_datasets : dict or None, optional
        Prepared observations organized by modality and dataset subset.
        Passed to the model; observation-based training is not yet
        implemented.
    collocation_dataset : dict or None, optional
        Prepared collocation mapping containing ``"xf"`` and ``"yf"``.
        The model currently requires this mapping even for supervised
        runs, where both values may be ``None``.

    Returns
    -------
    PhysicsInformedNN
        Initialized model wrapper.
    """

    model = PhysicsInformedNN(
        network, # MLP or GNN 
        params, # general parameters
        cfd_datasets=cfd_datasets,
        observation_datasets=observation_datasets,
        collocation_dataset=collocation_dataset,
    )

    return model
