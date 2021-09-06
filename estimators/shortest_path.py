"""
This code modified from:
* https://github.com/dgranata/Intrinsic-Dimension/blob/master/ID_fit.py

Original author comments below:
   "# Comments and questions can be send to daniele.granata@gmail.com

    #  In case you find the code useful, please cite
    #  Daniele Granata, Vincenzo Carnevale
    #  "Accurate estimation of intrinsic dimension using graph distances: unraveling the geometric complexity of datasets"
    #  Scientific Report, 6, 31377 (2016)
    #  https://www.nature.com/articles/srep31377"
"""
import sys, argparse
import os
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
# ppope: MODIFED TO USE FASTER VERSION OF SHORTEST PATH
# from sklearn.utils.graph import graph_shortest_path
from gsp.graph_shortest_path import graph_shortest_path

import torch
from torchvision import transforms
from .utils import KNNComputerNoCheck, update_nn


def func(x, a, b, c):
    return a * np.log(np.sin(x / 1 * np.pi / 2.))


def func2(x, a):
    return -a / 2. * (x - 1) ** 2


def func3(x, a, b, c):
    return np.exp(c) * np.sin(x / b * np.pi / 2.) ** a


def get_knn_dists(anchor_dataset, full_dataset, K, args):
    """Compute the KNN distances. Including a 0 for the sample itself

    :param anchor_dataset:
    :param full_dataset:
    :param K:
    :return:
    """
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=K + 1, cosine_dist=args.cosine_dist).cuda()

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=args.bsize, shuffle=False,
                                                num_workers=args.n_workers)
    bootstrap_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=args.bsize, shuffle=False,
                                                   num_workers=args.n_workers)

    # neighb = NearestNeighbors(n_neighbors=k2 + 1, n_jobs=1, algorithm='ball_tree').fit(X)
    # dist, ind = neighb.kneighbors(X)
    update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)
    nn_dists = nn_computer.min_dists.cpu().numpy().reshape(-1)
    nn_col_idxes = nn_computer.nn_indices.cpu().numpy()
    nn_row_idxes = np.repeat(np.arange(nn_col_idxes.shape[0]).reshape(-1, 1), K+1, axis=1)
    # get the full sparse matrix
    knn_dists_full = scipy.sparse.csr_matrix((nn_dists, (nn_row_idxes.reshape(-1), nn_col_idxes.reshape(-1))),
                                             shape=(len(anchor_dataset), len(full_dataset)))
    return knn_dists_full


def shortest_path(dataset, args):
    me=args.metric
    n_neighbors = args.n_neighbors
    radius=args.radius+0
    MSA=False
    n_bins = args.n_bins
    rmax=args.r_max
    mm=-10000

    # debugging ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    all_inds = np.arange(len(dataset))
    rand_inds = np.random.choice(all_inds, size=1000, replace=False)
    dataset = torch.utils.data.Subset(dataset, rand_inds)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    data_dim = dataset[0][0].numel()
    first_n = len(dataset) if args.first_n is None else args.first_n

    # 1 Computing geodesic distance on connected points of the input file and relative histogram
    knn_dists_full = get_knn_dists(anchor_dataset=dataset,
                                   full_dataset=dataset,
                                   K=n_neighbors, args=args)
    if radius > 0:
        knn_dists_full = knn_dists_full.multiply( (knn_dists_full < radius).astype(np.float32))
    else:
        radius = knn_dists_full.max()

    C = graph_shortest_path(knn_dists_full, directed=False, first_n=first_n)
    C = np.asmatrix(C, dtype=np.float32)
    connect = np.zeros(C.shape[0])
    conn = np.zeros(C.shape[0])
    for i in range(0, C.shape[0]):
        conn_points = np.count_nonzero(C[i])
        conn[i] = conn_points
        if conn_points > C.shape[0] / 2.:
            connect[i] = 1
        else:
            C[i] = 0

    if np.count_nonzero(connect) > C.shape[0] / 2.:
        print('Number of connected points:', np.count_nonzero(connect), '(', 100 * np.count_nonzero(connect) / C.shape[0], '% )')
    else:
        print('The neighbors graph is highly disconnected, increase K or Radius parameters')
        sys.exit(2)

    triu_mask = np.triu(np.ones(C.shape, dtype=np.bool)) & (C > 0)
    dist_list = C[triu_mask][-1]

    #dist_file = open(os.path.join(args.out_path, 'dist.dat'), "w")
    #for i in range(0, len(dist_list)):
    #    dist_file.write("%s " % ((dist_list[i])))
    #dist_file.close()

    h = np.histogram(dist_list, n_bins)
    dx = h[1][1] - h[1][0]

    #plt.figure(1)
    #plt.plot(h[1][0:n_bins] + dx / 2, h[0], 'o-', label='histogram')
    #plt.xlabel('r')
    #plt.ylabel('N. counts')
    #plt.legend()
    #plt.savefig(os.path.join(args.out_path, 'hist.pdf'))
    #distr_x = []
    #distr_y = []

    avg = np.mean(dist_list)
    std = np.std(dist_list)

    if rmax > 0:
        avg = rmax
        std = min(std, rmax)
        print('\nNOTE: You fixed r_max for the initial fitting, average will have the same value')
    else:
        mm = np.argmax(h[0])
        rmax = h[1][mm] + dx / 2

    if args.r_max == -1:
        print('\nNOTE: You forced r_max to the maximum of the distribution in the initial fitting, avoiding consistency checks with the average')
        avg = rmax
        std = min(std, rmax)

    if args.r_min >= 0:
        print('\nNOTE: You fixed r_min for the initial fitting: r_min = ', args.r_min)
    if args.r_min == -1:
        print('\nNOTE: You forced r_min to the standard procedure in the initial fitting')

    print('\nDistances Statistics:')
    print('Average, standard dev., n_bin, bin_size, r_max, r_NN_max:', avg, std, n_bins, dx, rmax, radius, '\n')
    # 1
    tmp = 1000000
    if (args.r_min >= 0):
        tmp = args.r_min
    elif (args.r_min == -1):
        tmp = rmax - std

    if (np.fabs(rmax - avg) > std + 2. * dx):
        print('ERROR: There is a problem with the r_max detection:'
              '       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)\n'
              '       or r_max and r_avg are too distant and you may consider to fix the first detection of r_max with option -M'
              '       or to change the neighbor parameter with (-r/-k)')

        #plt.show()
        sys.exit()
    elif (rmax <= min(radius + dx, tmp)):
        print('ERROR: There is a problem with the r_max detection, it is shorter than the largest distance in the neighbors graph.\n'
              '       You may consider to fix the first detection of r_max with option -M and/or the r_min with option -n to fix the fit range\n'
              '       or to decrease the neighbors parameter with (-r/-k). For example It is possible to enforce the standard fit range with \n'
              '       r_min=r_max-2*sigma running option "-n -1"')

        #plt.show()
        sys.exit()

    # 2 Finding actual r_max and std. dev. to define fitting interval [rmin;rM]
    distr_x = h[1][0:n_bins] + dx / 2
    distr_y = h[0][0:n_bins]
    # pdb.set_trace()

    res = np.empty(data_dim)
    left_distr_x = np.empty(n_bins)
    left_distr_y = np.empty(n_bins)

    left_distr_x = distr_x[
        np.logical_and(np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2.0), distr_y[:] > 0.000001)]
    left_distr_y = np.log(distr_y[np.logical_and(np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2.0),
                                               distr_y[:] > 0.000001)])

    if (left_distr_y.shape[0] < 4):
        print('ERROR: Too few datapoints to fit the distribution:')
        print('       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)')
        print('       or the distance distribution itself has some issue')
        #plt.show()
        print('R, Dfit, Dmin', 'ERROR3', '\n')
        sys.exit()

    coeff = np.polyfit(left_distr_x, left_distr_y, 2, full='False')
    a0 = coeff[0][0]
    b0 = coeff[0][1]
    c0 = coeff[0][2]

    rmax_old = rmax
    std_old = std
    rmax = -b0 / a0 / 2.0

    if (args.r_max > 0): rmax = args.r_max
    # if(args.r_max==-1) : rmax=avg   #to be used in future in case of problem with Ymax
    if a0 < 0 and np.fabs(rmax - rmax_old) < std_old / 2 + dx:
        std = np.sqrt(-1 / a0 / 2.)
    else:
        rmax = avg
        std = std_old

    left_distr_x = distr_x[
        np.logical_and(distr_y[:] > 0.000001, np.logical_and(distr_x[:] > rmax - std, distr_x[:] < rmax + std / 2. + dx))]
    left_distr_y = np.log(distr_y[np.logical_and(distr_y[:] > 0.000001, np.logical_and(distr_x[:] > rmax - std, distr_x[
                                                                                                             :] < rmax + std / 2. + dx))])

    if (left_distr_y.shape[0] < 4):
        print('ERROR: Too few datapoints to fit the distribution:')
        print(
            '       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)')
        print('       or the distance distribution itself has some issue')
        #plt.show()
        sys.exit()

    coeff = np.polyfit(left_distr_x, left_distr_y, 2, full='False')
    a = coeff[0][0]
    b = coeff[0][1]
    c = coeff[0][2]

    rmax_old = rmax
    std_old = std
    if a < 0.:
        rmax = -b / a / 2.
        std = np.sqrt(-1 / a / 2.)  # it was a0

    rmin = max(rmax - 2 * std - dx / 2, 0.)
    if (args.r_min >= 0):
        rmin = args.r_min
    elif (rmin < radius and args.r_min != -1):
        rmin = radius
        print('\nWARNING: For internal consistency r_min has been fixed to the largest distance (r_NN_max) in the neighbors graph.')
        print('         It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-n -1" ')
        print('         or you can use -n to manually define a desired value (Example: -n 0.1)\n')

    rM = rmax + dx / 4

    if (np.fabs(rmax - rmax_old) > std_old / 4 + dx):  # fit consistency check
        print('\nWARNING: The histogram is probably not smooth enough (you may try to change n_bin with -b), rmax is fixed to the value of first iteration\n')

        rmax = rmax_old
        a = a0
        b = b0
        c = c0

        if (args.r_min >= 0):
            rmin = args.r_min
        elif (rmin < radius and args.r_min != -1):
            rmin = radius
            print('\nWARNING2: For internal consistency r_min has been fixed to the largest distance in the neighbors graph (r_NN_max).')
            print('          It is possible to reset the standard definition of r_min=r_max-2*sigma running with option "-n -1" ')
            print('          or you can use -n to manually define a desired value (Example: -n 0.1)\n')
        rM = rmax + dx / 4
    # 2

    # 3 Gaussian Fitting to determine ratio R
    # pdb.set_trace()
    left_distr_x = distr_x[
                       np.logical_and(np.logical_and(distr_x[:] > rmin, distr_x[:] <= rM), distr_y[:] > 0.000001)] / rmax
    left_distr_y = np.log(
        distr_y[np.logical_and(np.logical_and(distr_x[:] > rmin, distr_x[:] <= rM), distr_y[:] > 0.000001)]) - (
                               4 * a * c - b ** 2) / 4. / a

    if (left_distr_y.shape[0] < 4):
        print('ERROR: Too few datapoints to fit the distribution:')
        print(
            '       usually either the histogram is not smooth enough (you may consider changing the n_bins with option -b)')
        print('       or the distance distribution itself has some issue')
        #plt.show()
        sys.exit()

    fit = curve_fit(func2, left_distr_x, left_distr_y)
    ratio = np.sqrt(fit[0][0])
    y1 = func2(left_distr_x, fit[0][0])
    # 3

    # 4 Geodesics D-Hypersphere Distribution Fitting to determine Dfit

    fit = curve_fit(func, left_distr_x, left_distr_y)
    Dfit = (fit[0][0]) + 1

    y2 = func(left_distr_x, fit[0][0], fit[0][1], fit[0][2])
    # 4

    # 5 Determination of Dmin

    #D_file = open(os.path.join(args.out_path, 'D_residual.dat'), "w")

    for D in range(1, data_dim + 1):
        y = (func(left_distr_x, D - 1, 1, 0))
        for i in range(0, len(y)):
            res[D - 1] = np.linalg.norm((y) - (left_distr_y)) / np.sqrt(len(y))
        #D_file.write("%s " % D)
        #D_file.write("%s\n" % res[D - 1])

    Dmin = np.argmax(-res) + 1
    y = func(left_distr_x, Dmin - 1, fit[0][1], 0)
    # 5

    # 6 Printing results
    print('\nFITTING PARAMETERS:')
    print('rmax, std. dev., rmin', rmax, std, rmin)
    print('\nFITTING RESULTS:')
    print('R, Dfit, Dmin', ratio, Dfit, Dmin, '\n')
    results = {'R': float(ratio), 'Dfit': float(Dfit), 'Dmin': int(Dmin)}
    return results
