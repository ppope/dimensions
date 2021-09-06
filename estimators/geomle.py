"""
This file modified from:
* https://github.com/stat-ml/GeoMLE/blob/master/geomle/geomle.py
"""

import numpy as np
import pandas as pd
from functools import partial

import torch
from .utils import KNNComputerNoCheck, update_nn
from sklearn.linear_model import LinearRegression, Ridge


def mle_center(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation

    Input parameters:
    X        - data points
    X_center - data centers
    k        - number of nearest neighbours (Default = 5)
    dist     - matrix of distances to the k nearest neighbors of each point (Optional)

    Returns:
    dimensionality estimation for the k
    """
    dist = dist[:, 0:k]
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    Rs = dist[:, -1]

    return intdim_sample, Rs


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
    dist = dist[:, 0:k]
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    inv_mle = d.copy()
    d = 1. / d
    mle = d
    Rs = dist[:, -1]
    inv_Rs = 1. / dist[:, -1]
    return mle, inv_mle, Rs, inv_Rs


def tolist(x):
    if type(x) in {int, float}:
        return [x]
    if type(x) in {list, tuple}:
        return list(x)
    if type(x) == np.ndarray:
        return x.tolist()

def fit_poly_reg(x, y, w=None, degree=(1, 2), alpha=5e-3):
    """
    Fit regression and return f(0)

    Input parameters:
    x - features (1d-array)
    y - targets (1d-array)
    w - weights for each points (Optional)
    degree - degrees of polinoms (Default tuple(1, 2))
    alpha - parameter of regularization (Default 5e-3)
    Returns:
    zero coefficiend of regression
    """
    X = np.array([x ** i for i in tolist(degree)]).T
    lm = Ridge(alpha=alpha)
    lm.fit(X, y, w)
    return lm.intercept_


def _func(df, degree, alpha, inv_mle=False):
    gr_df = df.groupby('k')
    if inv_mle:
        d = gr_df['inv_mle_dim'].mean().values
        std = gr_df['inv_mle_dim'].std().values
        R = gr_df['inv_mle_R'].mean().values
    else:
        d = gr_df['dim'].mean().values
        std = gr_df['dim'].std().values
        R = gr_df['R'].mean().values
    if np.isnan(std).any(): std = np.ones_like(std)
    return fit_poly_reg(R, d, std**-1, degree=degree, alpha=alpha)

def drop_zero_values(dist):
    mask = dist[:,0] == 0
    dist[mask] = np.hstack([dist[mask][:, 1:], dist[mask][:,0:1]])
    dist = dist[:, :-1]
    assert np.all(dist > 0)
    return dist


def geomle(full_dataset, k1=10, k2=40, nb_iter1=10, nb_iter2=20, degree=(1, 2),
           alpha=5e-3, ver='GeoMLE', random_state=None, debug=False, args=None):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples

    Input parameters:
    X            - data
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 40)
    nb_iter1     - number of bootstrap iterations (Default = 10)
    nb_iter1     - number of bootstrap iterations for each regresion (Default = 20)
    degree       - (Default = (1, 2))
    alpha        - (Default = 5e-3)
    random_state - random state (Optional)

    Returns:
    array of shape (nb_iter1,) of regression dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = len(full_dataset)
    dim_space = float("inf")

    result = []
    data_reg = []
    for i in range(nb_iter1):
        dim_all, R_all, k_all, idx_all = [], [], [], []
        inv_dim_all = []
        inv_R_all = []
        for j in range(nb_iter2):
            idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
            X_bootstrap = torch.utils.data.Subset(full_dataset, idx)
            #print("Bootstrap round {} with {} samples".format(j, len(X_bootstrap)))
            # compute the KNN with pytorch
            nn_computer = KNNComputerNoCheck(len(full_dataset), K=k2+1).cuda()

            anchor_loader = torch.utils.data.DataLoader(full_dataset,
                                                        batch_size=args.bsize, shuffle=False,
                                                        num_workers=args.n_workers)
            bootstrap_loader = torch.utils.data.DataLoader(X_bootstrap,
                                                        batch_size=args.bsize, shuffle=False,
                                                        num_workers=args.n_workers)

            update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)

            dist = nn_computer.min_dists.cpu().numpy()
            dist = drop_zero_values(dist)
            dist = dist[:, :k2]
            assert np.all(dist > 0)

            for k in range(k1, k2 + 1):
                # if args.inv_mle:
                #     dim, inv_mle_dim, R, R_inv = intrinsic_dim_sample_wise_double_mle(k=k, dist=dist)
                #     inv_dim_all += inv_mle_dim.tolist()
                #     inv_R_all += R_inv.tolist()
                # else:
                dim, R = mle_center(k, dist)
                dim_all += dim.tolist()
                R_all += R.tolist()
                idx_all += list(range(nb_examples))
                k_all += [k] * dim.shape[0]
            #print("Finished bootstrap {}".format(j))
        data = {'dim': dim_all,
                'R': R_all,
                'idx': idx_all,
                'k': k_all}
        # if args.inv_mle:
        #     data['inv_mle_dim'] = inv_dim_all
        #     data['inv_mle_R'] = inv_R_all

        df = pd.DataFrame(data)
        if ver.lower() == 'GeoMLE'.lower():
            func = partial(_func, degree=degree, alpha=alpha, inv_mle=False)
            # if args.inv_mle:
            #     reg = 1. / (1. / df.groupby('idx').apply(func).values).mean()
            # else:
            reg = df.groupby('idx').apply(func).values
            if args.inv_mle:
                reg = 1. / (1. / reg).mean()
            else:
                reg = reg.mean()
            data_reg.append(df)
        elif ver.lower() == 'fastGeoMLE'.lower():
            df_gr = df.groupby(['idx', 'k']).mean()[['R', 'dim']]
            R = df_gr.groupby('k').R.mean()
            d = df_gr.groupby('k').dim.mean()
            std = df_gr.groupby('k').dim.std()
            reg = fit_poly_reg(R, d, std ** -1, degree=degree, alpha=alpha)
            data_reg.append((R, d, std))
        else:
            assert False, 'Unknown mode {}'.format(ver)
        reg = 0 if reg < 0 else reg
        # reg = dim_space if reg > dim_space else reg
        result.append(reg)
    if debug:
        return np.array(result), data_reg
    else:
        return np.array(result)
