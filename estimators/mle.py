"""
This file modified from:
* https://github.com/stat-ml/GeoMLE/blob/master/geomle/mle.py
"""

import torch
import random
import numpy as np
from estimators.utils import KNNComputerNoCheck, update_nn
from sklearn.neighbors import NearestNeighbors


def intrinsic_dim_sample_wise_double_mle(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani

    Input parameters:
    X    - data
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k (or more) nearest neighbors of each point (Optional)

    Returns:
    two dimensionality estimates
    """
    dist = dist[:, 1:(k + 1)]
    if not np.all(dist > 0):
        # trying to catch the bug
        np.save("error_dist.npy", dist)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    inv_mle = d.copy()

    d = 1. / d
    mle = d
    return mle, inv_mle


def intrinsic_dim_sample_wise(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation

    Input parameters:
    X    - data
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k nearest neighbors of each point (Optional)

    Returns:
    dimensionality estimation for the k
    """
    dist = dist[:, 1:(k + 1)]
    if not np.all(dist > 0):
        raise Exception("Identical samples detected!")
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample


def intrinsic_dim_scale_interval(k1=10, k2=20, dist=None):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2, k1 < k2

    Input parameters:
    X    - data
    k1   - minimal number of nearest neighbours (Default = 10)
    k2   - maximal number of nearest neighbours (Default = 20)
    dist - matrix of distances to the k nearest neighbors of each point (Optional)

    Returns:
    list of Levina-Bickel dimensionality estimation for k = k1..k2
    """
    intdim_k = []

    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(k, dist).mean()
        intdim_k.append(m)
    return intdim_k


def mle(full_dataset, nb_iter=100, random_state=None, k1=10, k2=20, average=False, args=None):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples

    Input parameters:
    X            - data
    nb_iter      - number of bootstrap iterations (Default = 100)
    random_state - random state (Optional)
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 20)
    average      - if False returns array of shape (nb_iter, k2-k1+1) of the estimations for each bootstrap samples (Default = True)

    Returns:
    array of shape (k2-k1+1,) of Levina-Bickel dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)

    if args.anchor_samples > 0:
        indices = [i for i in range(len(full_dataset))]
        random.shuffle(indices)
        subset_idxes = indices[:args.anchor_samples]
        anchor_dataset = torch.utils.data.Subset(full_dataset, subset_idxes)
        nb_examples = len(anchor_dataset)
    else:
        anchor_dataset = full_dataset
        nb_examples = len(full_dataset)
    results = []

    print("Computing the KNNs")
    # compute the KNN with pytorch
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=k2 + 1).cuda()

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=args.bsize, shuffle=False,
                                                num_workers=args.n_workers)
    bootstrap_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=args.bsize, shuffle=False,
                                                   num_workers=args.n_workers)

    dist = nn_computer.min_dists.cpu().numpy()

    Rs = []
    for i in range(k1, k2 + 1):
        Rs.append(np.max(dist[:, :i]))

    for i in range(nb_iter):
        idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
        #print("Bootstrap round {} with {} samples".format(i, len(idx)))

        results.append(intrinsic_dim_scale_interval(k1, k2, dist[idx, :]))
    results = np.array(results)

    if average:
        return results.mean(axis=0), Rs
    else:
        return results, Rs


def mle_inverse_singlek(full_dataset, k1=10, args=None, anchor_dataset=None):
    """
    Returns the Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani

    Input parameters:
    X            - data
    k1           - minimal number of nearest neighbours (Default = 10)
    average      - if False returns array of shape (nb_iter, k2-k1+1) of the estimations for each bootstrap samples (Default = True)

    Returns:
    two dimensionality estimates
    """

    if args.anchor_ratio > 0:
        args.anchor_samples = int(args.anchor_ratio * len(full_dataset))

    if anchor_dataset is None:
        if args.anchor_samples > 0:
            print("Using {} anchor samples. ".format(args.anchor_samples))
            indices = [i for i in range(len(full_dataset))]
            random.shuffle(indices)
            subset_idxes = indices[:args.anchor_samples]
            anchor_dataset = torch.utils.data.Subset(full_dataset, subset_idxes)
        else:
            anchor_dataset = full_dataset

    print("Computing the KNNs")
    # compute the KNN with pytorch
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=k1 + 1).cuda()

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=args.bsize, shuffle=False,
                                                num_workers=args.n_workers)
    bootstrap_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=args.bsize, shuffle=False,
                                                   num_workers=args.n_workers)

    # neighb = NearestNeighbors(n_neighbors=k2 + 1, n_jobs=1, algorithm='ball_tree').fit(X)
    # dist, ind = neighb.kneighbors(X)
    update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)
    dist = nn_computer.min_dists.cpu().numpy()

    if args.eval_every_k:
        mle_res, inv_mle_res = [], []
        for k in range(2, k1+1):
            mle_results, invmle_results = intrinsic_dim_sample_wise_double_mle(k, dist)
            mle_res.append(mle_results.mean())
            inv_mle_res.append(1. / invmle_results.mean())

        return mle_res, inv_mle_res
    else:
        mle_results, invmle_results = intrinsic_dim_sample_wise_double_mle(k1, dist)

        return mle_results.mean(), 1. / invmle_results.mean()


def mle_inverse_singlek_loop(full_dataset, net, k1=5, k2=15, k_step=5, average=False, args=None):
    """
    Returns the correction of MLE by MacKay-Ghahramani

    Input parameters:
    X            - data
    net          - network
    k1           - minimum number of nearest neighbours (Default = 5)
    k2           - maximum number of nearest neighbours (Default = 15)
    k_step       - step size to loop from k1 to k2 (Default = 5)
    average      - to take the average of all estimates

    Returns:
    array of dimensionality estimates, or an averaged estimate
    """

    if args.anchor_ratio > 0:
        args.anchor_samples = int(args.anchor_ratio * len(full_dataset))

    if args.anchor_samples > 0:
        print("Using {} anchor samples. ".format(args.anchor_samples))
        indices = [i for i in range(len(full_dataset))]
        random.shuffle(indices)
        subset_idxes = indices[:args.anchor_samples]
        anchor_dataset = torch.utils.data.Subset(full_dataset, subset_idxes)
    else:
        anchor_dataset = full_dataset

    print("Computing the KNNs")
    # compute the KNN with pytorch
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=k2 + 1).cuda()

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=args.bsize, shuffle=False,
                                                num_workers=args.n_workers)
    bootstrap_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=args.bsize, shuffle=False,
                                                   num_workers=args.n_workers)

    update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)
    dist = nn_computer.min_dists.cpu().numpy()

    inv_mle_res = []
    for k in range(k1, k2+1, k_step):
        mle_results, invmle_results = intrinsic_dim_sample_wise_double_mle(k, dist)
        inv_mle_res.append(1. / invmle_results.mean())

    if average:
        return inv_mle_res.mean()
    else:
        return inv_mle_res
