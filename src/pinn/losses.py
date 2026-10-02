# SPDX-License-Identifier: MIT
"""Compute CFD, observation, physics, and validation losses."""

import torch

from .residuals import (
    grad,
    steady_euler_residuals,
    steady_compressible_rans_residuals,
)

def _zero_loss(pinn):
    """Create a zero loss on the model's device.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model providing the target device.

    Returns
    -------
    torch.Tensor
        Scalar float32 zero on ``pinn.device``.
    """

    return torch.tensor(0.0, dtype=torch.float32, device=pinn.device)

def _cfd_loss_terms(pinn, x, y, rho_true, u_true, v_true, p_true,
                   mut_true=None, use_dropout=False, role="data"):
    """Compute mean-squared errors for CFD flow fields.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model used to predict the flow fields.
    x, y : torch.Tensor
        Coordinates at which to compare predictions and targets.
    rho_true, u_true, v_true, p_true : torch.Tensor
        Target nondimensional density, velocity components, and pressure.
    mut_true : torch.Tensor or None, optional
        Target scaled turbulent viscosity, required for RANS.
    use_dropout : bool, optional
        Enable data dropout. Defaults to False.
    role : str or None, optional
        GNN graph role. Defaults to ``"data"``; ignored by the MLP.

    Returns
    -------
    tuple of torch.Tensor
        Scalar losses ``(l_rho, l_u, l_v, l_p, l_mut)``.
        The viscosity loss is zero for Euler.
    """

    if pinn.eq == 'euler': 
        rho_pred, u_pred, v_pred, p_pred = \
            pinn.net_fields(x, y, use_dropout, role=role)        
        # is not rans
        l_mut = _zero_loss(pinn)

    elif pinn.eq == 'rans':
        rho_pred, u_pred, v_pred, p_pred, mut_pred \
            = pinn.net_fields(x, y, use_dropout, role=role)

        if mut_true is None:
            raise ValueError("For RANS, mut_t must be provided in loss_fn.")
        # Loss of the turbulent viscosity         
        l_mut = torch.mean((mut_true - mut_pred) ** 2)
        
    else:
        raise ValueError(f"Unknown equation type: {pinn.eq}")

    # CFD losses terms
    l_rho = torch.mean((rho_true - rho_pred) ** 2)
    l_u   = torch.mean((u_true   - u_pred)   ** 2)
    l_v   = torch.mean((v_true   - v_pred)   ** 2)
    l_p   = torch.mean((p_true   - p_pred)   ** 2)

    return l_rho, l_u, l_v, l_p, l_mut

def _observations_loss_terms(pinn, observations, use_dropout=False, 
                            role="data"):
    """Compute mean-squared errors for observed flow quantities.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model providing predictions and the schlieren density gradient 
        type.
    observations : dict
        Optional entries: ``schlieren``, ``velocity_u``, ``velocity_v``, 
        and ``pressure_taps``. Each contains coordinate tensor ``X`` of 
        shape ``(N, 2)`` and target tensor ``value`` with N values.
    use_dropout : bool, optional
        Enable data dropout. Defaults to False.
    role : str or None, optional
        GNN graph role. Defaults to ``"data"``; ignored by the MLP.

    Returns
    -------
    tuple of torch.Tensor
        Scalar losses ``(l_obs_sch, l_obs_u, l_obs_v, l_obs_p)``.
        Missing observation entries contribute zero loss.
    """

    if "schlieren" in observations:
        schlieren = observations["schlieren"]

        # Coordiantes
        x_obs_sch = schlieren["X"][:,0:1].detach().requires_grad_(True)
        y_obs_sch = schlieren["X"][:,1:2].detach().requires_grad_(True)

        # Value
        # Both tensors now represent d(rho*) / d(x*)
        sch_obs_true = schlieren["value"].reshape(-1,1)
        rho_obs_pred = pinn.net_fields(
            x_obs_sch, y_obs_sch, use_dropout, role=role
        )[0]
        
        # Calculate density gradient - schlieren
        if pinn.schlieren_grad_type == "grad_x":
            sch_obs_pred = grad(rho_obs_pred, x_obs_sch)
 
        elif pinn.schlieren_grad_type == "grad_y":
            sch_obs_pred = grad(rho_obs_pred, y_obs_sch)

        elif pinn.schlieren_grad_type == "magnitude":
            drho_dx = grad(rho_obs_pred, x_obs_sch)
            drho_dy = grad(rho_obs_pred, y_obs_sch)

            eps = 1e-8 
            sch_obs_pred = torch.sqrt(
                drho_dx.square() + drho_dy.square() + eps**2
            )

        else:
            raise ValueError(
                f"Unkown schlieren gradient type: {pinn.schlieren_grad_type}"
            )

        # schlieren loss
        l_obs_sch = torch.mean((sch_obs_true - sch_obs_pred) ** 2)

    else:
         l_obs_sch = _zero_loss(pinn)

    if "pressure_taps" in observations:
        pressure_taps = observations["pressure_taps"]

        # Coordiantes
        x_obs_p = pressure_taps["X"][:,0:1]
        y_obs_p = pressure_taps["X"][:,1:2]

        # Value
        p_obs_true = pressure_taps["value"].reshape(-1,1)
        p_obs_pred = pinn.net_fields(
            x_obs_p, y_obs_p, use_dropout, role=role
        )[3]        

        # pressure taps loss
        l_obs_p = torch.mean((p_obs_true - p_obs_pred) ** 2)

    else:
        l_obs_p = _zero_loss(pinn)

    if "velocity_u" in observations:
        velocity_u = observations["velocity_u"]

        # Coordiantes
        x_obs_u = velocity_u["X"][:,0:1]
        y_obs_u = velocity_u["X"][:,1:2]

        # Value
        u_obs_true = velocity_u["value"].reshape(-1,1)
        u_obs_pred = pinn.net_fields(
            x_obs_u, y_obs_u, use_dropout, role=role
        )[1]        

        # u-velocity loss
        l_obs_u = torch.mean((u_obs_true - u_obs_pred) ** 2)

    else:
        l_obs_u = _zero_loss(pinn)

    if "velocity_v" in observations:
        velocity_v = observations["velocity_v"]

        # Coordiantes
        x_obs_v = velocity_v["X"][:,0:1]
        y_obs_v = velocity_v["X"][:,1:2]

        # Value
        v_obs_true = velocity_v["value"].reshape(-1,1)
        v_obs_pred = pinn.net_fields(
             x_obs_v, y_obs_v, use_dropout, role=role
        )[2]        

        # v-velocity loss
        l_obs_v = torch.mean((v_obs_true - v_obs_pred) ** 2)

    else:
        l_obs_v = _zero_loss(pinn)

    return l_obs_sch, l_obs_u, l_obs_v, l_obs_p

def _residual_loss_terms(pinn):
    """Compute mean-squared physics residuals.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model with collocation points and Euler or RANS parameters.

    Returns
    -------
    tuple of torch.Tensor
        Scalar losses ``(l_f1, l_f2, l_f3, l_f4)`` for mass, x-momentum,
        y-momentum, and energy. All four are zero for supervised models.
    """

    if pinn.model == 'supervised':
        z = _zero_loss(pinn)
        return z, z, z, z
    
    if pinn.model != 'pinn':
        raise ValueError(f"Unknown model type: {pinn.model}")

    if pinn.xf is None or pinn.yf is None:
        raise ValueError("PINN mode requires collocation points xf and yf.")

    # Need gradients wrt x,y
    # Independent coordinate tensors for automatic differentiation
    x = pinn.xf.clone().detach().requires_grad_(True)
    y = pinn.yf.clone().detach().requires_grad_(True)

    # For the GNN, net_fields() inserts these collocation
    # coordinates into the complete training graph.
    if pinn.net_arch == "mlp":
        role = None

    elif pinn.net_arch == "gnn":
        role = "residual"

    else:
        raise ValueError(
            f"Unknown network architecture: {pinn.net_arch}"
        )

    # Residuals for each equation
    if pinn.eq == 'euler':
        rho_pred, u_pred, v_pred, p_pred = \
            pinn.net_fields(x, y, use_dropout=False, role=role)        

        f1_res, f2_res, f3_res, f4_res \
            = steady_euler_residuals(pinn, x, y, rho_pred, u_pred, v_pred, p_pred)
        
    elif pinn.eq == 'rans':
        rho_pred, u_pred, v_pred, p_pred, mut_pred = \
            pinn.net_fields(x, y, use_dropout=False, role=role)                
        
        f1_res, f2_res, f3_res, f4_res \
            = steady_compressible_rans_residuals(pinn, x, y, rho_pred, u_pred, \
                                                 v_pred, p_pred, mut_pred)

    else:
        raise ValueError(f"Unknown equation type: {pinn.eq}")
        
    # PDE residual loss terms
    l_f1  = torch.mean(f1_res ** 2)
    l_f2  = torch.mean(f2_res ** 2)
    l_f3  = torch.mean(f3_res ** 2)
    l_f4  = torch.mean(f4_res ** 2)

    return l_f1, l_f2, l_f3, l_f4

def loss_fn(pinn, return_terms=False):
    """Compute weighted data and physics training losses.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing training data, collocation points, and loss 
        weights.
    return_terms : bool, optional
        Include individual loss terms for logging. Defaults to False.

    Returns
    -------
    loss : torch.Tensor
        Total weighted loss: ``data_loss + res_loss``.
    data_loss : torch.Tensor
        Weighted CFD and observation losses combined.
    res_loss : torch.Tensor
        Weighted physics residual loss.
    detached_terms : tuple of torch.Tensor, optional
        Returned as a fourth item only when ``return_terms`` is True.
        Contains detached, unweighted terms in this order:
        ``(l_rho, l_u, l_v, l_p, l_mut, l_obs_sch, l_obs_u, l_obs_v,
        l_obs_p, l_f1, l_f2, l_f3, l_f4)``.
    """
    
    # Define dropout
    use_data_dropout = pinn.enable_data_dropout

    # CFD loss terms
    if pinn.problem == "forward" or pinn.use_inv_boundaries:
        l_rho, l_u, l_v, l_p, l_mut = \
            _cfd_loss_terms(pinn, pinn.xtrain, pinn.ytrain, pinn.rhotrain, 
                            pinn.utrain, pinn.vtrain, pinn.ptrain, 
                            pinn.muttrain, use_data_dropout, role="data")
    else:
        # if not forward or not bc from inverse
        l_rho, l_u, l_v, l_p, l_mut = (
            _zero_loss(pinn) for _ in range(5)
        )

    # Obeservation loss terms
    l_obs_sch, l_obs_u, l_obs_v, l_obs_p = \
        _observations_loss_terms(pinn, pinn.obs_train, use_data_dropout, 
                                 role="data")

    # Residuals loss terms
    l_f1, l_f2, l_f3, l_f4 = _residual_loss_terms(pinn)

    # CFD loss
    cfd_loss = (
        pinn.w_rho * l_rho +
        pinn.w_u   * l_u   +
        pinn.w_v   * l_v   +
        pinn.w_p   * l_p   +
        pinn.w_mut * l_mut
    )
    # Observations loss
    obs_loss = (
        pinn.w_obs_sch * l_obs_sch +
        pinn.w_obs_u * l_obs_u +
        pinn.w_obs_v * l_obs_v +
        pinn.w_obs_p * l_obs_p 
    )
    # Residual loss
    res_loss = (
        pinn.w_f1 * l_f1 + 
        pinn.w_f2 * l_f2 + 
        pinn.w_f3 * l_f3 + 
        pinn.w_f4 * l_f4
    )

    # Data loss
    data_loss = cfd_loss + obs_loss

    # Total loss: data loss + residual loss
    loss = data_loss + res_loss  

    if return_terms:
        terms = (
            l_rho, l_u, l_v, l_p, l_mut,
            l_obs_sch, l_obs_u, l_obs_v, l_obs_p,
            l_f1, l_f2, l_f3, l_f4,
        )

        # Logging values do not need their computation graphs.
        detached_terms = tuple(term.detach() for term in terms)

        return loss, data_loss, res_loss, detached_terms

    return loss, data_loss, res_loss

def validation_loss_fn(pinn):
    """Compute validation loss without dropout, restoring the model's mode.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing validation data and loss weights.

    Returns
    -------
    torch.Tensor or None
        Detached, weighted CFD and observation loss, or ``None`` when
        validation is unavailable.
    """

    if not pinn.has_validation:
        return None

    # Save the current PyTorch module mode.
    was_training = pinn.training
    # Use the model in prediction/evaluation mode
    pinn.eval()

    try:
        validation_loss = _zero_loss(pinn)

        if pinn.has_cfd_validation:
            with torch.no_grad():
                # CFD loss terms
                l_val_rho, l_val_u, l_val_v, l_val_p, l_val_mut = (
                    _cfd_loss_terms(
                        pinn,
                        pinn.xval,
                        pinn.yval,
                        pinn.rhoval,
                        pinn.uval,
                        pinn.vval,
                        pinn.pval,
                        pinn.mutval,
                        use_dropout=False,
                        role="validation"
                    )
                )

                # CFD loss
                cfd_loss = (
                    pinn.w_rho * l_val_rho +
                    pinn.w_u   * l_val_u   +
                    pinn.w_v   * l_val_v   +
                    pinn.w_p   * l_val_p   +
                    pinn.w_mut * l_val_mut
                )

            validation_loss = validation_loss + cfd_loss

        if pinn.has_observation_validation:
            with torch.enable_grad():
            # Observation loss terms
                l_obs_sch, l_obs_u, l_obs_v, l_obs_p = (
                    _observations_loss_terms(
                        pinn,
                        pinn.obs_val,
                        use_dropout=False,
                        role="validation"
                    )
                )

                # Observations loss
                observation_loss =  (
                    pinn.w_obs_sch * l_obs_sch +
                    pinn.w_obs_u * l_obs_u +
                    pinn.w_obs_v * l_obs_v +
                    pinn.w_obs_p * l_obs_p
                )

            validation_loss = validation_loss + observation_loss.detach()

    finally:
        # Restore the model's original mode
        pinn.train(was_training)

    return validation_loss.detach()
