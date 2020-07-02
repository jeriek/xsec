"""
This module evaluates cross-sections for the production of
supersymmetric particles, using pre-trained Distributed Gaussian
Processes.
"""

from __future__ import print_function
import warnings

import numpy as np

import xsec.utils as utils
import xsec.parameters as parameters
import xsec.gploader as gploader
import xsec.features as features

###############################################
# Evaluation functions                        #
###############################################


def eval_xsection(verbose=2, check_consistency=True):
    """
    Evaluate cross-sections for processes in global list
    gploader.PROCESSES using parameter values stored in global
    dictionary parameters.PARAMS.

    Parameters
    ----------
    verbose : int, optional
         Degree of verbosity when printing results to screen:
        - verbose=0 : prints nothing
        - verbose=1 : prints a single line per process (no header),
            in the following format:
              PID1 PID2 xsection_central regdown_rel regup_rel scaledown_rel
              scaleup_rel pdfdown_rel pdfup_rel alphasdown_rel alphasup_rel
        - verbose=2 : prints full description of the result (default)
    check_consistency : bool, optional
        If True (default), forces a consistency check of the parameters
        in PARAMS. This function checks that all necessary parameters
        have been set, that they are internally consistent and that the
        parameters are in a range where the evaluation can be trusted.

    Returns
    -------
    return_array : array, shape (9, n_processes)
        Numpy array containing results of the evaluation, with format
          [xsection_central, regdown_rel, regup_rel, scaledown_rel,
           scaleup_rel, pdfdown_rel, pdfup_rel, alphasdown_rel, alphasup_rel]
        where each component is an array with one number for each process;
        the order of the processes is that in the global list PROCESSES.

        The returned quantities are defined as follows:
        - xsection_central :
            central-scale xsection in femtobarn (fb)
        - regdown_rel, regup_rel :
            signed regression errors divided by xsection_central
        - scaledown_rel, scaleup_rel :
            signed scale errors (from varying the scale to 0.5x and 2x
            the central scale) divided by xsection_central
        - pdfdown_rel, pdfup_rel :
            signed PDF errors divided by xsection_central
        - alphasdown_rel, alphasup_rel :
            signed (symmetrized) alpha_s errors divided by xsection_central

        Note: All the *down_rel uncertainties are given a negative sign
    """

    ##################################################
    # Build feature vector                           #
    ##################################################

    # Get local variable to avoid multiple slow lookups in global namespace
    processes = gploader.PROCESSES
    params = parameters.PARAMS

    # Sanity check parameter inputs
    if check_consistency:
        # First make a list of all unique features used for all processes
        feature_list = features.get_unique_features(processes)
        parameters.check_parameters(feature_list)

    # Build feature vectors, depending on production channel
    process_features = features.get_features_dict(processes)

    ###################################################
    # Do DGP evaluation  (all hard work happens here) #
    ###################################################

    # Call a DGP for each process_xstype, store results as lists of
    # (mu_dgp, sigma_dgp) in dictionary with key xstype; i-th element of
    # list dgp_results[xstype] gives DGP result for
    # process = processes[i] and the specified xstype.
    # Immediately corrected for any data transformation during training.

    # Dictionary of PROCESSES-ordered lists
    dgp_results = {
        xstype: [
            gploader.TRANSFORM_MODULES[(process, xstype)].inverse_transform(
                process,
                xstype,
                params,
                *dgp_predict(process, xstype, process_features[process])
            )
            for process in processes
        ]
        for xstype in utils.XSTYPES
    }

    ###################################################
    # Constructing the right output                   #
    ###################################################

    # All returned errors are signed RELATIVE errors (dimensionless).
    # The absolute errors are obtained by multiplying with the central-
    # scale cross section.

    # -- Central-scale xsection and regression error (= standard
    #    deviation or asymmetric error) in fb.
    xsection_central, reg_err = map(
        np.array,
        list(zip(*(mu_sigma_dgp for mu_sigma_dgp in dgp_results["centr"]))),
    )

    # (zip() splits list of (mu,sigma) tuples into two tuples, one for
    # mu and one for sigma values -- then convert to arrays by mapping)
    # NOTE: Result arrays are ordered in the user-specified order from
    # the global PROCESSES variable!

    # -- Signed regression errors on the central cross section, divided
    #    by xsection_central
    # --- Each element of reg_err is a [lower, upper] error array
    regdown_rel = -reg_err[:, 0] / xsection_central
    regup_rel = reg_err[:, 1] / xsection_central

    # -- Signed scale errors (from varying the scale to 0.5x and 2x the
    #    central scale) divided by xsection_central. To prevent that the
    #    unusual case with xsection_scaleup > xsection_scaledown causes
    #    errors, min/max ensures scaledown_rel always gives the lower
    #    bound and scaleup_rel the higher one.
    #    NOTE: This means scaledown_rel generally doesn't correspond to
    #    the xsection value at the lower scale, but at the higher one,
    #    and vice versa for scaleup_rel.
    # --- Get the DGP medians, discard regression errors on the variations
    mu_dgp_scldn_rel, _ = np.array(
        list(zip(*dgp_results["scldn"])), dtype=object
    )
    mu_dgp_sclup_rel, _ = np.array(
        list(zip(*dgp_results["sclup"])), dtype=object
    )

    scaledown_rel = (
        np.array(
            list(map(np.min, list(zip(mu_dgp_scldn_rel, mu_dgp_sclup_rel)))),
            dtype=object,
        )
        - 1.0
    )
    scaleup_rel = (
        np.array(
            list(map(np.max, list(zip(mu_dgp_scldn_rel, mu_dgp_sclup_rel)))),
            dtype=object,
        )
        - 1.0
    )

    # -- Signed PDF errors divided by xsection_central
    # --- Get the DGP medians, discard regression errors on the variations
    delta_pdf_rel, _ = np.array(list(zip(*dgp_results["pdf"])), dtype=object)

    pdfdown_rel = (-1.0) * np.abs(delta_pdf_rel)
    pdfup_rel = np.abs(delta_pdf_rel)

    # -- Signed alpha_s errors divided by xsection_central.
    #    Following the PDF4LHC recommendation in arxiv:1510.03865,
    #    we take the cross-section uncertainty coming from alpha_s
    #    variation to be half the range between the cross-sections
    #    calculated with upper and lower (1 sigma) alpha_s values.
    #    (Therefore this cross-section uncertainty is symmetric.)
    # --- Get the DGP medians, discard regression errors on the variations
    mu_dgp_adn_rel, _ = np.array(list(zip(*dgp_results["adn"])), dtype=object)
    mu_dgp_aup_rel, _ = np.array(list(zip(*dgp_results["aup"])), dtype=object)

    delta_alphas_rel = np.array(
        [
            0.5 * (abs(aup_rel - adn_rel))
            for (aup_rel, adn_rel) in list(zip(mu_dgp_aup_rel, mu_dgp_adn_rel))
        ]
    )

    alphasdown_rel = (-1.0) * delta_alphas_rel
    alphasup_rel = delta_alphas_rel

    # Collect values for output in Numpy array
    return_array = np.array(
        [
            xsection_central,
            regdown_rel,
            regup_rel,
            scaledown_rel,
            scaleup_rel,
            pdfdown_rel,
            pdfup_rel,
            alphasdown_rel,
            alphasup_rel,
        ]
    )

    # Print result to screen, depending on verbosity level
    utils.print_result(return_array, processes, verbose)

    return return_array


def dgp_predict(process, xstype, new_features):
    """
    Evaluate a set of distributed Gaussian processes (DGPs). The DGP
    'experts' are combined according to the Generalized Robust
    Bayesian Committee Machine (GRBCM) algorithm, arXiv:1806.00720.

    Parameters
    ----------
    process : tuple
        Two-component tuple of integers specifying the final state.
        For example, gluino-gluino production corresponds to the tuple
        (1000021, 1000021).
    xstype : str
        Cross-section type to be computed, chosen from the list in
        utils.XSTYPES.
    new_features : list of float
        List with values for the required features of the chosen
        process (for the new test point), as constructed by the function
        features.get_features_dict().

    Returns
    -------
    mu_dgp : float
        Central value following the GRBCM prescription.
    sigma_dgp : float
        Uncertainty estimate following the GRBCM prescription.
    """
    assert len(process) == 2
    process_xstype = utils.get_process_id(process, xstype)

    # Find the number of trained experts for the given process
    n_experts = len(gploader.PROCESS_DICT[process_xstype])

    # Empty arrays where all predicted numbers are stored
    mus = np.empty(n_experts)
    sigmas = np.empty(n_experts)

    # Loop over GP experts (the first is the communications expert)
    for i in range(n_experts):
        mu, sigma = gp_predict(
            process_xstype, new_features, index=i + 1, return_std=True
        )
        mus[i] = mu
        sigmas[i] = sigma

    # Shorthand variables for the 'communication expert' (i=0)
    mu_c = mus[0]
    sigma_c = sigmas[0]

    # Find weight (beta) for each expert
    betas = np.ones(n_experts)
    for i in range(n_experts):
        if i < 2:
            betas[i] = 1.0
        else:
            betas[i] = 0.5 * (2.0 * np.log(sigma_c) - 2.0 * np.log(sigmas[i]))

    # If any of the betas is negative (sigma_c < sigma_i), due to worse
    # fit despite more points, set weight of that expert to zero.
    # In principle this cannot occur when the experts share hyperparameters,
    # but roundoff errors can still cause it.
    if any(betas < 0):
        # warnings.warn(
        #     "Warning: one or more weights beta_i were negative and set to 0."
        #     "\n -- weights: {w} for {p} - {t}".format(
        #         w=betas, p=process, t=xstype
        #     )
        # )
        betas = np.maximum(betas, 0.0)

    # Final mean and variance
    if n_experts == 1:
        var_dgp_inv = sigma_c ** (-2)
        mu_dgp = mu_c
    else:
        var_dgp_inv = np.sum(betas[1:] * sigmas[1:] ** (-2)) - (
            np.sum(betas[1:]) - 1.0
        ) * sigma_c ** (-2)
        mu_dgp = var_dgp_inv ** (-1) * (
            np.sum(betas[1:] * sigmas[1:] ** (-2) * mus[1:])
            - (np.sum(betas[1:]) - 1.0) * sigma_c ** (-2) * mu_c
        )

    # Return mean and std (_not_ variance!)
    sigma_dgp = np.sqrt(var_dgp_inv ** (-1))

    return mu_dgp, sigma_dgp


def gp_predict(
    process_xstype, new_features, index=1, return_std=True, return_cov=False
):
    """
    Gaussian process evaluation for the individual experts. Takes as
    input arguments the produced partons, an array of new test features,
    and the index number of the expert. Requires running
    load_processes() first.

    Returns a list of Numpy arrays containing the central value and the
    GP uncertainty.

    Based on Algorithm 2.1 of Gaussian Processes for Machine Learning by
    Rasmussen and Williams (MIT Press, 2006). The structure of this
    function is inspired by GaussianProcessRegressor.predict() from
    scikit-learn v0.19.2.

    Parameters
    ----------
    process_xstype : tuple
        The input argument process_xstype is a 3-tuple
        (process[0], process[1], var) where the first two components
        are integers specifying the process and the last component is a
        string from utils.XSTYPES.
        Example: (1000021, 1000021, 'centr')
    new_features : list of float
        List with values for the required features of the chosen
        process (for the new test point), as constructed by the function
        features.get_features_dict().
    index : int, optional
        Index of the GP expert, starting from index 1 (default) for the
        communications expert.
    return_std : bool, optional
        Whether to return the standard deviation of the GP posterior
        predictive distribution (default: True).
    return_cov : bool, optional
        Whether to return the  of the full covariance matrix of the GP
        (default: False). This option is currently unavailable.
    Returns
    -------
    y_mean : float
        Mean value of the Gaussian GP posterior predictive distribution.
    y_std : float
        Standard deviation of the Gaussian GP posterior predictive
        distribution.
    """

    if return_std and return_cov:
        raise RuntimeError(
            "Cannot return both standard deviation and full covariance."
        )

    # Get dict of loaded models for the specified process, and select
    # the GP expert indicated by the index (i=1 is the communications expert)
    try:
        gp_model = gploader.PROCESS_DICT[process_xstype][index]
        if gploader.USE_CACHE:
            gp_model = gp_model.get()

        X_train = gp_model["X_train"]
        alpha = gp_model["alpha"]
        # L_inv = gp_model["L_inv"]  # Only required for return_cov
        K_inv = gp_model["K_inv"]
        kernel = gploader.KERNEL_DICT[(process_xstype, index)]

    except KeyError:
        raise KeyError(
            "No trained GP models loaded for: {id}".format(id=process_xstype)
        )

    X = np.atleast_2d(new_features)

    # Transpose of K*
    K_trans = kernel(X, X_train)
    # Line 4 (y_mean = f_star)
    y_mean = K_trans.dot(alpha)

    # Prior variance: 1x1 if just 1 new test point!
    prior_variance = kernel(X)

    if return_std:
        # Compute variance of predictive distribution. Note: equal to
        # prior_variance if 1x1, deep copy else prior_variance is y_var
        # alias.
        y_var = np.diag(prior_variance.copy())
        y_var.setflags(write=True)  # else this array is read-only
        K_trans_dot_K_inv = np.dot(K_trans, K_inv)
        y_var -= np.einsum(
            "ij,ij->i", K_trans_dot_K_inv, K_trans, optimize=True
        )

        # Check if any of the variances is negative because of numerical
        # issues. If yes: set the variance to absolute value, to keep
        # the rough order of magnitude right.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted some variance(s) smaller than 0."
                " Approximating these with their absolute value."
            )
            print("Numerical precision error: negative variance computed!")
            y_var[y_var_negative] = np.abs(y_var[y_var_negative])

        # Add uncertainty due to fitting the hyperparameters, here the case
        # with constant mean (estimated as sample mean), arXiv:1606.03865
        # Effects become large when far* from training points (* in length
        # scale units)
        g = 1.0 - np.sum(K_trans_dot_K_inv)
        M_inv = np.sum(K_inv) ** (-1)
        y_var += M_inv * g ** 2

        # Return the standard deviation
        y_std = np.sqrt(y_var)
        # prior_std = np.sqrt(prior_variance.flatten())
        return y_mean, y_std

    elif return_cov:
        warnings.warn(
            "The return_cov option in gp_predict has been disabled "
            "to decrease memory usage."
        )
        # v = L_inv.dot(K_trans.T)  # Line 5
        # y_cov = prior_variance - K_trans.dot(v)  # Line 6
        # return y_mean, y_cov, prior_variance
        return y_mean

    else:
        return y_mean
