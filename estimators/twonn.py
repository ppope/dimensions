import torch
import numpy as np
from .utils import KNNComputerNoCheck, update_nn


def twonn(full_dataset, args=None):
    """
    Returns the (fractional) TwoNN estimate of intrinsic dimension (ID)

    Input parameters:
    X            - data

    Returns:
    The TwoNN ID estimate
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
    # compute the 2-NN with pytorch
    nn_computer = KNNComputerNoCheck(len(anchor_dataset), K=2+1).cuda()

    anchor_loader = torch.utils.data.DataLoader(anchor_dataset,
                                                batch_size=args.bsize, shuffle=False,
                                                num_workers=args.n_workers)
    bootstrap_loader = torch.utils.data.DataLoader(full_dataset,
                                                   batch_size=args.bsize, shuffle=False,
                                                   num_workers=args.n_workers)

    update_nn(anchor_loader, 0, bootstrap_loader, 0, nn_computer)
    dist = nn_computer.min_dists.cpu().numpy()

    print("Computing TwoNN regression")
    n = len(anchor_dataset)
    mu = np.zeros(n)
    for i in range(n):
      mu[i] = dist[i, 1+1] / dist[i, 0+1]

    mu = np.sort(mu)
    mu = mu[0:n-1]
    x = np.log(mu)
    empF = np.arange(1, n) / n
    y = -np.log(1.0 - empF)

    # Force intercept to 0
    A = np.c_[x, np.zeros(n-1)]
    slope, _ = np.linalg.lstsq(A, y)[0]

    return slope
