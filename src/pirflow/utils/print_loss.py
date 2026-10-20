# SPDX-License-Identifier: MIT

def print_loss(pinn, it, loss_val, terms):
    """Print the current training-loss components.

    Parameters
    ----------
    pinn : PhysicsInformedNN
        Model containing data and physical loss terms.
    it : int
        Training iteration.

    Returns
    -------
    None
        Losses are written to standard output.
    """
    
    # Defer the import to avoid circular imports.
    from ..pinn.losses import loss_fn

    (
        l_rho, l_u, l_v, l_p, l_mut, 
        l_obs_sch, l_obs_u, l_obs_v, l_obs_p,
        l_f1, l_f2, l_f3, l_f4 
    ) = terms

    print(
        f"It: {it:6d} | "
        f"weighted total loss: {loss_val.item():.3e} | ",
        end="",
    )        

    if pinn.model == "pinn":

        print(
            f"res_1: {l_f1.item():.3e} | "
            f"res_2: {l_f2.item():.3e} | "
            f"res_3: {l_f3.item():.3e} | "
            f"res_4: {l_f4.item():.3e} | ",
            end="", 
        )

    if pinn.problem == "forward":
        
        print(
            f"cfd_rho: {l_rho.item():.3e} | "
            f"cfd_u: {l_u.item():.3e} | "
            f"cfd_v: {l_v.item():.3e} | "
            f"cfd_p: {l_p.item():.3e} | ",
            end="", 
        )

        if pinn.eq == 'rans':

            print(
                f"cfd_mut: {l_mut.item():.3e} | ",
                end="",
            )

    elif pinn.problem == "inverse":

        if pinn.use_inv_boundaries:
            boundary_terms = {
                "rho": l_rho, "u": l_u, "v": l_v, "p": l_p,
            }
            if pinn.eq == "rans":
                boundary_terms["mut"] = l_mut

            for field, term in boundary_terms.items():
                if (
                    pinn.observed_train is not None
                    and not pinn.observed_train[field].any().item()
                ):
                    continue
                print(
                    f"bc_{field}: {term.item():.3e} | ", end="",
                )

        if pinn.schlieren_enabled:

            print(
                f"obs_sch: {l_obs_sch.item():.3e} | ",
                end="", 
            )

        if pinn.velocity_profiles_enabled:

            print(
                f"obs_u: {l_obs_u.item():.3e} | "
                f"obs_v: {l_obs_v.item():.3e} | ",
                end="", 
            )

        if pinn.pressure_taps_enabled:

            print(
                f"obs_p: {l_obs_p.item():.3e} | ",
                end="", 
            )

    else:
         raise ValueError(
              f"Unknown run.problem: {pinn.problem}"
         )

    print("")
