import os
import json
import argparse
import random
import numpy as np

import torch
import torchvision

import estimators
from data.dataloader import load_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--bsize', default=1024, type=int,
                        help='batch size for previous images')
    parser.add_argument('--estimator', default="mle", type=str, choices=['mle','geomle','twonn', 'shortest-path'])
    parser.add_argument('--n-workers', default=1, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--save-path', default='results.json',
                        type=str, help='path to store the results')

    #Estimator-specific args
    ##MLE args
    parser.add_argument('--k1', default=20, type=int)
    parser.add_argument('--k2', default=55, type=int)
    parser.add_argument('--nb-iter', default=100, type=int)
    parser.add_argument('--eval-every-k', default=False, action="store_true",
                        help="Whether to evaluate every k<=k1")
    parser.add_argument('--average-inverse', default=False, action="store_true",
                        help='Whether to take the average of the inverse from each bootstrap run ')
    parser.add_argument('--single-k', default=False, action="store_true",
                        help='Whether to estimate the dimension with a single k')
    ##GeoMLE args
    parser.add_argument('--nb-iter1', default=1, type=int)
    parser.add_argument('--nb-iter2', default=20, type=int)
    parser.add_argument('--inv-mle', default=False, action='store_true')

    ##Shortest-Path Args
    parser.add_argument("-m", "--metric", type=str,
                        help="define the scipy distance to be used   (Default: euclidean or hamming for MSA)",
                        default='euclidean')
    parser.add_argument("-x", "--matrix",
                        help="if the input file contains already the complete upper triangle of a distance matrix (2 Formats: (idx_i idx_j distance) or simply distances list ) (Opt)",
                        action="store_true")
    parser.add_argument("-k", "--n_neighbors", type=int, help="nearest_neighbors parameter (Default k=3)", default=3)
    parser.add_argument("-r", "--radius", type=float, help="use neighbor radius instead of nearest_neighbors  (Opt)",
                        default=0.)
    parser.add_argument("-b", "--n_bins", type=int, help="number of bins for distance histogram (Default 50)",
                        default=50)
    parser.add_argument("-M", "--r_max", type=float,
                        help="fix the value of distance distribution maximum in the fit (Opt, -1 force the standard fit, avoiding consistency checks)",
                        default=0)
    parser.add_argument("-n", "--r_min", type=float,
                        help="fix the value of shortest distance considered in the fit (Opt, -1 force the standard fit, avoiding consistency checks)",
                        default=-10)
    parser.add_argument("-D", "--direct", help="analyze the direct (not graph) distances (Opt)", action="store_true")
    parser.add_argument("-I", "--projection", help="produce an Isomap projection using the first ID components (Opt)",
                   action="store_true")
    parser.add_argument('--cosine-dist', default=False, action='store_true')
    parser.add_argument('--first-n', default=None, type=int)

    #Dataset args
    parser.add_argument('--dset', default="cifar10", type=str)
    parser.add_argument('--anchor-samples', default=0, type=int,
                        help="0 for using all samples from the training set")
    parser.add_argument('--anchor-ratio', default=0, type=float,
                        help="0 for using all samples from the training set")
    parser.add_argument('--max_num_samples', default=-1, type=int,
                        help="Maximum number of samples to process." \
                             "Useful for evaluating convergence.")
    parser.add_argument('--imagenet-dir', default="/fs/cml-datasets/ImageNet/ILSVRC2012/train", type=str)
    parser.add_argument('--n_cls', default=None, type=int,
                        help="A redundant flag for specifying number of classes.")
    parser.add_argument('--separate-classes', default=False, action="store_true")

    args, _ = parser.parse_known_args()
    return args


def set_gpu(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_estimator(args, dataset):
    estimator = args.estimator
    if estimator == "mle":
        results = run_mle(args, dataset)
    elif estimator == "geomle":
        results = run_geomle(args, dataset)
    elif estimator == "twonn":
        results = run_twonn(args, dataset)
    elif estimator == "shortest-path":
        results = run_shortest_path(args, dataset)
    return results


def run_mle(args, dataset):
    if args.single_k:
        dim, inv_mle_dim = estimators.mle_inverse_singlek(dataset, k1=args.k1, args=args)
    else:
        if args.average_inverse:
            indiv_est = mle(dataset, net, nb_iter=args.nb_iter, random_state=None, k1=args.k1, k2=args.k2, average=True, args=args)[0]
            dim = 1. / np.mean(1. / indiv_est)
        else:
            dim = mle(dataset, net, nb_iter=args.nb_iter, random_state=None, k1=args.k1, k2=args.k2, average=True, args=args)[0].mean()
        inv_mle_dim = None

    #Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    if args.eval_every_k:
        for nk, k in enumerate(range(2, args.k1+1)):
            save_dict["k{}_dim".format(k)] = float(dim[nk])
            print("k={}, Estimated dimension of inv mle: {}".format(k, inv_mle_dim[nk]))
            save_dict["k{}_inv_mle_dim".format(k)] = float(inv_mle_dim[nk])
    else:
        save_dict["dim"] = float(dim)
        if inv_mle_dim is not None:
            print("Estimated dimension of inv mle: {}".format(inv_mle_dim))
            save_dict["inv_mle_dim"] = float(inv_mle_dim)
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


def run_geomle(args, dataset):
    dim_ = estimators.geomle(dataset, k1=args.k1, k2=args.k2, nb_iter1=args.nb_iter1, nb_iter2=args.nb_iter2,
                 degree=(1, 2), alpha=5e-3, ver='GeoMLE', random_state=None, debug=False, args=args)
    dim = dim_.mean()
    print("Estimated dimension: {}".format(dim))
    #Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


def run_twonn(args, dataset):
    dim = estimators.twonn(dataset, args=args)
    print("Estimated dimension: {}".format(dim))
    #Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


def run_shortest_path(args, dataset):
    results = estimators.shortest_path(dataset, args=args)
    dim = results['Dmin']
    print("Estimated dimension: {}".format(dim))
    #Log and save results
    save_fp = args.save_path
    save_dict = vars(args)
    save_dict["dim"] = dim
    save_dict["results"] = results
    with open(save_fp, 'w') as fh:
        json.dump(save_dict, fh)
    return save_dict


if __name__ == "__main__":
    args = parse_args()
    set_gpu(args.gpu)
    set_seed(args.seed)
    dataset = load_data(args)
    run_estimator(args, dataset)
