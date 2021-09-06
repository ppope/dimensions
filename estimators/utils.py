import torch
from torch import nn
from torch.utils.data.sampler import Sampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random


def get_curvature(num_samples, dists, fig_name="curvature"):
    N = np.log(num_samples)
    eps = np.log(dists)
    m, b = np.polyfit(eps, N, 1)
    print(m, b)
    fig = plt.figure()
    plt.scatter(eps, N)
    fig.suptitle('CIFAR10', fontsize=12)
    plt.xlabel('log(eps)', fontsize=12)
    plt.ylabel('log(N)', fontsize=12)
    plt.savefig("{}_m{}_b{}.pdf".format(fig_name, m, b))
    # plt.show()


class KNNComputer(nn.Module):
    """
    Using this hack for data parallel
    """
    def __init__(self, sample_num, K=1):
        super(KNNComputer, self).__init__()

        self.K = K
        self.register_buffer("num_computed", torch.zeros([]))
        if K == 1:
            self.register_buffer("min_dists", torch.full((sample_num, ), float('inf')))
        else:
            self.register_buffer("min_dists", torch.full((sample_num, K), float('inf')))

    def forward(self, x, x_idx_start, y, y_idx_start):
        # update the min dist for existing examples...
        x_bsize, y_bsize = x.size(0), y.size(0)
        x = x.view(x_bsize, -1)
        y = y.view(y_bsize, -1)
        # dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)
        dist = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        # need to ignore the distance to the data point itself
        x_idxes = torch.Tensor([i for i in range(x_idx_start, x_idx_start+x_bsize)]).view(-1, 1).to(x)
        y_idxes = torch.Tensor([i for i in range(y_idx_start, y_idx_start+y_bsize)]).view(1, -1).to(x)
        ignore_idx_mask = x_idxes == y_idxes
        if torch.any(ignore_idx_mask):
            dist[ignore_idx_mask] = float('inf')

        if self.K == 1:
            new_min_dist, _ = torch.min(dist, dim=1)

            self.min_dists[x_idx_start:x_idx_start+x_bsize] = torch.min(new_min_dist,
                                                                  self.min_dists[x_idx_start:x_idx_start+x_bsize])
        else:
            comp = torch.cat([dist, self.min_dists[x_idx_start:x_idx_start+x_bsize]], dim=1)
            updated_min_dist, _ = torch.topk(comp, self.K, dim=1, largest=False)
            self.min_dists[x_idx_start:x_idx_start + x_bsize] = updated_min_dist

    def get_mean_nn_dist(self, sidx, eidx):
        if self.K == 1:
            return torch.mean(self.min_dists[sidx:eidx])


class KNNComputerNoCheck(nn.Module):
    """
    Using this hack for data parallel
    without checking for the sample itself
    """
    def __init__(self, sample_num, K=1, cosine_dist=False):
        super(KNNComputerNoCheck, self).__init__()

        self.K = K
        self.cosine_dist = cosine_dist
        self.register_buffer("num_computed", torch.zeros([]))

        if K == 1:
            self.register_buffer("min_dists", torch.full((sample_num, ), float('inf')))
            self.register_buffer("nn_indices", torch.full((sample_num,), 0, dtype=torch.int64))
        else:
            self.register_buffer("min_dists", torch.full((sample_num, K), float('inf')))
            self.register_buffer("nn_indices", torch.full((sample_num, K), 0, dtype=torch.int64))

    def forward(self, x, x_idx_start, y, y_idx_start):
        # update the min dist for existing examples...
        x_bsize, y_bsize = x.size(0), y.size(0)
        x = x.view(x_bsize, -1)
        y = y.view(y_bsize, -1)
        if self.cosine_dist:

            x = x / x.norm(dim=1, keepdim=True)
            y = y / y.norm(dim=1, keepdim=True)
            dist = x.mm(y.t())

        else:
            # dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)
            dist = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        if self.K == 1:
            new_min_dist, nn_idxes = torch.min(dist, dim=1)

            self.min_dists[x_idx_start:x_idx_start + x_bsize] = torch.min(new_min_dist,
                                                                  self.min_dists[x_idx_start:x_idx_start+x_bsize])

            self.nn_indices[x_idx_start:x_idx_start + x_bsize] = nn_idxes + y_idx_start
        else:
            comp = torch.cat([dist, self.min_dists[x_idx_start:x_idx_start+x_bsize]], dim=1)
            # updated_min_dist, nn_idxes = torch.topk(comp, self.K, dim=1, largest=False)
            # check for repeated images
            sorted_dists, sorted_idxes = torch.sort(comp, dim=1, descending=False)
            updated_dist_list, nn_idx_list = [], []
            for row in range(sorted_dists.shape[0]):
                sidx = 1
                while sidx < sorted_dists.shape[1]:
                    if sorted_dists[row, sidx] == 0:
                        sidx += 1
                    else:
                        break
                updated_dist_list.append(sorted_dists[row, sidx-1:sidx-1+self.K])
                nn_idx_list.append(sorted_idxes[row, sidx-1:sidx-1+self.K])
            updated_min_dist = torch.stack(updated_dist_list)
            nn_idxes = torch.stack(nn_idx_list)

            self.min_dists[x_idx_start:x_idx_start + x_bsize] = updated_min_dist

            sample_idxes = (nn_idxes < y_bsize).int() * (nn_idxes + y_idx_start) \
                           + (nn_idxes >= y_bsize).int() * self.nn_indices[x_idx_start:x_idx_start + x_bsize]
            self.nn_indices[x_idx_start:x_idx_start + x_bsize] = sample_idxes

    def get_mean_nn_dist(self, sidx, eidx):
        if self.K == 1:
            return torch.mean(self.min_dists[sidx:eidx])


def update_nn(anchor_loader, anchor_start_idx, new_img_loader, new_start_idx, nn_computer):
    anchor_counter = anchor_start_idx
    # ignoring the labels
    with torch.no_grad():
        for n, (abatch, _) in enumerate(anchor_loader):
            abatch = abatch.cuda()

            new_img_counter = new_start_idx
            for newbatch, _ in new_img_loader:
                newbatch = newbatch.cuda()

                nn_computer(abatch, anchor_counter, newbatch, new_img_counter)

                new_img_counter += newbatch.size(0)

                equiv_flag = (nn_computer.min_dists[anchor_start_idx:anchor_start_idx+abatch.size(0), 0] == 0) & (nn_computer.min_dists[anchor_start_idx:anchor_start_idx+abatch.size(0), 1] == 0)
                if torch.any(equiv_flag):
                    raise Exception("Identical data detected!")

            anchor_counter += abatch.size(0)

            #if n % 50 == 0 or n == len(anchor_loader) - 1:
            #    #print("Finished {} images".format(anchor_counter))


def create_random_subsets(data_set, subset_size):
    indices = [i for i in range(len(data_set))]
    random.shuffle(indices)

    n_subsets = len(data_set) // subset_size
    subset_idxes = [indices[i*subset_size: (i+1)*subset_size] for i in range(n_subsets)]

    return [torch.utils.data.Subset(data_set, sidxes) for sidxes in subset_idxes], indices


if __name__ == "__main__":
    a = np.array([[1000, 2712.478024044418],
                  [2000, 2619.277852208197],
                  [3000, 2589.1015581140155],
                  [1000, 2682.8301552900048],
                  [2000, 2620.22951004717],
                  [3000, 2586.9299704307186],
                  [4000, 2554.141911272333],
                  [5000, 2535.0052768383703],
                  [6000, 2522.2212855240937],
                  [7000, 2511.0527067369403],
                  [8000, 2502.671622266195],
                  [9000, 2495.539064381195],
                  [10000, 2486.039414654384],
                  [11000, 2477.7142354562543],
                  [12000, 2472.932534098392],
                  [13000, 2467.1793436622843],
                  [14000, 2460.917650877651],
                  [15000, 2456.5837979002963],
                  [16000, 2452.474002175604],
                  [17000, 2446.8778746473686],
                  [18000, 2440.2417276528345],
                  [19000, 2438.1349311383183],
                  [20000, 2432.8156051204332],
                  [21000, 2428.0422095658582],
                  [22000, 2421.7713682349595],
                  [23000, 2417.6188608773787],
                  [24000, 2413.5819662049053],
                  [25000, 2409.63769064015],
                  [26000, 2404.552513391291],
                  [27000, 2400.8546459362456],
                  [28000, 2397.870976128168],
                  [29000, 2396.4524131333696],
                  [30000, 2393.4408500046475],
                  [31000, 2391.2262810675757],
                  [32000, 2387.5649130287675],
                  [33000, 2385.33937431289],
                  [34000, 2382.713900003717],
                  [35000, 2381.3509362320515],
                  [36000, 2377.9903364974234],
                  [37000, 2374.980190342692],
                  [38000, 2373.0929623133998],
                  [39000, 2370.9822093349585],
                  [40000, 2368.499529460011],
                  [41000, 2367.2265188939014],
                  [42000, 2364.8408086144873],
                  [43000, 2362.7624684435928],
                  [44000, 2360.7418029887267],
                  [45000, 2358.6602399524786],
                  [46000, 2357.559231063025],
                  [47000, 2355.2225615862767],
                  [48000, 2353.5068017056165],
                  [49000, 2351.510501062334],
                  [50000, 2349.7408100451203]])
    N = np.log(a[:, 0])
    eps = np.log(a[:, 1])
    m, b = np.polyfit(eps, N, 1)
    print(m, b)
    fig = plt.figure()
    plt.scatter(eps, N)
    fig.suptitle('CIFAR10', fontsize=12)
    plt.xlabel('log(eps)', fontsize=12)
    plt.ylabel('log(N)', fontsize=12)
    plt.show()
