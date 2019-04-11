"""
This module evaluates cross-sections for the production of
supersymmetric particles, using pre-trained Distributed Gaussian
Processes.
"""

from __future__ import print_function

import numpy as np

import xsec.utils as utils
import xsec.gploader as gploader
import xsec.parameters as parameters
import xsec.features as features
import xsec.kernels as kernels


###############################################
# Evaluation functions                        #
###############################################


def eval_xsection(verbose=2, check_consistency=True):
    """
    Evaluates cross sections for processes in global list PROCESSES
    using parameter values stored in global dictionary PARAMS.

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
        - regdown_rel, regup_rel:
            xsection deviating one (lognormal) regression error down/up
            from the central-scale xsection, relative to the latter
        - scaledown_rel, scaleup_rel :
            lower/higher xsection from varying the scale (to 0.5x and 2x
            the central scale), relative to the central-scale xsection
        - pdfdown_rel, pdfup_rel :
            xsection deviating one PDF error down/up from the
            central-scale xsection, relative to the latter
        - alphasdown_rel, alphasup_rel :
            xsection deviating one (symmetrised) alpha_s error down/up
            from the central-scale xsection, relative to the latter

        Note: All returned errors are defined to be deviations from 1.
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

    # All returned errors are defined to be deviations from 1

    # -- Central-scale xsection and regression error (= standard
    #    deviation) in fb.
    xsection_central, reg_err = map(
        np.array, zip(*(mu_sigma_dgp for mu_sigma_dgp in dgp_results["centr"]))
    )
    # xsection_central, reg_err = map(
    #     np.array, zip(*(moments_lognormal(*mu_sigma_dgp)
    #                     for mu_sigma_dgp in dgp_results['centr']))
    #     )
    # (zip() splits list of (mu,sigma) tuples into two tuples, one for
    # mu and one for sigma values -- then convert to arrays by mapping)
    # NOTE: Result arrays are now ordered in the user-specified order
    # from the global PROCESSES variable!

    # -- Xsection deviating one (lognormal) regression error away
    #    from the central-scale xsection, relative to the latter.
    regdown_rel = 1.0 - reg_err / xsection_central  # numpy array
    regup_rel = 1.0 + reg_err / xsection_central  # numpy array

    # -- Xsection at lower and higher scale (0.5x and 2x central scale),
    #    relative to the central-scale xsection. To prevent that the
    #    unusual case with xsection_scaleup > xsection_scaledown causes
    #    errors, min/max ensures scaledown_rel always gives the lower
    #    bound and scaleup_rel the higher one.
    #    NOTE: This means scaledown_rel generally doesn't correspond to
    #    the xsection value at the lower scale, but at the higher one,
    #    and vice versa for scaleup_rel.
    # Get the DGP means, discard regression errors on the variations
    mu_dgp_scldn, _ = np.array(zip(*dgp_results["scldn"]))
    mu_dgp_sclup, _ = np.array(zip(*dgp_results["sclup"]))

    scaledown_rel = np.array(map(np.min, zip(mu_dgp_scldn, mu_dgp_sclup)))
    scaleup_rel = np.array(map(np.max, zip(mu_dgp_scldn, mu_dgp_sclup)))

    # -- Xsection deviating one PDF error away from the
    #    central-scale xsection, relative to the latter.
    # Get the DGP means, discard regression errors on the variations
    delta_pdf_rel, _ = np.array(zip(*dgp_results["pdf"]))

    pdfdown_rel = 1.0 - delta_pdf_rel
    pdfup_rel = 1.0 + delta_pdf_rel

    # -- Xsection deviating one symmetrised alpha_s error away from
    #    the central-scale xsection, relative to the latter.
    # Get the DGP means, discard regression errors on the variations
    mu_dgp_adn, _ = np.array(zip(*dgp_results["scldn"]))
    mu_dgp_aup, _ = np.array(zip(*dgp_results["sclup"]))

    delta_alphas_rel = np.array(
        [
            0.5 * (abs(aup - 1.0) + abs(1.0 - adn))
            for (aup, adn) in zip(mu_dgp_aup, mu_dgp_adn)
        ]
    )

    alphasdown_rel = 1.0 - delta_alphas_rel
    alphasup_rel = 1.0 + delta_alphas_rel

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
    utils.print_result(return_array, verbose)

    return return_array


def dgp_predict(process, xstype, features):
    """
        Evaluate a set of distributed Gaussian processes (DGPs). The DGP
        'experts' are combined according to the Generalized Robust
        Bayesian Committee Machine algoritm, arxiv:1806.00720.
    """
    assert len(process) == 2
    process_xstype = utils.get_process_id(process, xstype)

    # Find the number of trained experts for the given process
    n_experts = len(gploader.PROCESS_DICT[process_xstype])

    # Empty arrays where all predicted numbers are stored
    mus = np.zeros(n_experts)
    sigmas = np.zeros(n_experts)

    # Loop over GP experts
    for i in range(n_experts):
        mu, sigma, sigma_prior = gp_predict(
            process_xstype, features, index=i, return_std=True
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
            betas[i] = 0.5 * (2 * np.log(sigma_c) - 2 * np.log(sigmas[i]))

    # Final mean and variance:
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
    process_xstype, features, index=0, return_std=True, return_cov=False
):
    """
    Gaussian process evaluation for the individual experts. Takes as
    input arguments the produced partons, an array of new test features,
    and the index number of the expert. Requires running
    load_processes() first.

    Returns a list of numpy arrays containing the mean value (the
    predicted cross-section), the GP standard deviation (or full
    covariance matrix), and the square root of the prior variance on the
    test features.

    Based on Algorithm 2.1 of Gaussian Processes for Machine Learning by
    Rasmussen and Williams (MIT Press, 2006). The structure of this
    function is inspired by GaussianProcessRegressor.predict() from
    scikit-learn v0.19.2.
    """

    if return_std and return_cov:
        raise RuntimeError(
            "Cannot return both standard deviation and full covariance."
        )

    try:
        if gploader.USE_CACHE:
            # Get list of loaded models for the specified process
            gp_model = gploader.PROCESS_DICT[process_xstype].get()[index]
        else:
            gp_model = gploader.PROCESS_DICT[process_xstype][index]

        kernel = gp_model["kernel"]
        X_train = gp_model["X_train"]
        alpha = gp_model["alpha"]
        L_inv = gp_model["L_inv"]
        K_inv = gp_model["K_inv"]
        kernel = kernels.get_kernel(gp_model["kernel"])

    except KeyError:
        raise KeyError(
            "No trained GP models loaded for: {id}".format(process_xstype)
        )

    X = np.atleast_2d(features)

    K_trans = kernel(X, X_train)  # transpose of K*
    y_mean = K_trans.dot(alpha)  # Line 4 (y_mean = f_star)

    prior_variance = kernel(X)  # Note: 1x1 if just 1 new test point!]

    if return_std:
        # Compute variance of predictive distribution. Note: equal to
        # prior_variance if 1x1, deep copy else prior_variance is y_var
        # alias.
        y_var = np.diag(prior_variance.copy())
        y_var.setflags(write=True)  # else this array is read-only
        y_var -= np.einsum(
            "ij,ij->i", np.dot(K_trans, K_inv), K_trans, optimize=True
        )

        # Check if any of the variances is negative because of numerical
        # issues. If yes: set the variance to absolute value, to keep
        # the rough order of magnitude right.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            # warnings.warn("Predicted some variance(s) smaller than 0."
            #  " Approximating these with their absolute value.")
            y_var[y_var_negative] = np.abs(y_var[y_var_negative])
        y_std = np.sqrt(y_var)
        prior_std = np.sqrt(prior_variance.flatten())
        return y_mean, y_std, prior_std

    elif return_cov:
        v = L_inv.dot(K_trans.T)  # Line 5
        y_cov = prior_variance - K_trans.dot(v)  # Line 6
        return [y_mean, y_cov, prior_variance]

    else:
        return y_mean
