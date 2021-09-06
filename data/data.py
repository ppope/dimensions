import gc
import os
import h5py
import json
import numpy as np
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FileDataset(Dataset):
    def __init__(self, input_dir, max_num_samples, label=0, **kwargs):
        #Load config
        fp = os.path.join(input_dir, "config.json")
        with open(fp, 'r') as fh:
            config = json.load(fh)
        self.label = label
        self.batch_size = config['batch_size']
        batch_fns = sorted(glob(os.path.join(input_dir, "*.pt")))
        if max_num_samples != -1:
            #Adjust number of samples
            max_num_batches = max_num_samples // self.batch_size
            num_batches = len(batch_fns)
            if num_batches > max_num_batches:
                batch_fns = batch_fns[:max_num_batches]

        self.batch_fns = batch_fns
        self.num_batches = len(self.batch_fns)
        self.num_samples = self.num_batches * self.batch_size
        self.targets = np.repeat(label, self.num_samples).astype('int')

        self.compute_stats()

    def compute_stats(self):
        """
        Compute data mean and variance in online manner.

        Data may be too large to load all at once.

        Reference
        * http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

        """
        #Compute mean
        current_mean = 0
        current_var = 0
        m = 0
        for b in range(self.num_batches):
            self.update_batch(b)
            batch_mean = self.batch.mean()
            batch_var = self.batch.var()
            n = self.batch.shape[0]
            new_mean = (m/(m+n))*current_mean + (n/(m+n))*batch_mean
            new_var  = (  (m/(m+n))*current_var
                        + (n/(m+n))*batch_var
                        + (m*n/(m+n)**2)*(current_mean - batch_mean)**2  )
            m += n
            current_mean = new_mean
            current_var = new_var
        self.mean = current_mean
        self.var = current_var
        self.std = torch.sqrt(self.var)
        self.num_samples = m

    def update_batch(self, batch_num):
        self.current_batch_num = batch_num
        batch = torch.load(self.batch_fns[self.current_batch_num])
        self.batch = batch.type(torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        #Compute batch index:
        batch_num = index // self.batch_size
        #Load new batch if not in current one
        if batch_num != self.current_batch_num:
            self.update_batch(batch_num)
        ind = index - batch_num*self.batch_size
        target_ = self.targets[index]
        target = torch.Tensor([target_])
        sample = self.batch[ind]
        #Standardize
        #NB: Implicit cast to float32 here if data is uint8
        #This is intended.
        sample = (sample - self.mean) / self.std
        return (sample, target)


class TensorDataset(Dataset):
    """TensorDataset with support of transforms.

       Modfied from:
          https://stackoverflow.com/a/55593757
    """
    def __init__(self, tensors, transform=None):
        X,Y = tensors
        self.X = X
        self.targets = Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        #NB: X is mutable
        self.num_samples = len(self.X)
        return self.num_samples


def read_hdf5(filepath, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.
    :param filepath: path to file to read
    :type filepath: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray

    TAKEN FROM:
      * https://github.com/davidstutz/disentangling-robustness-generalization/blob/master/common/utils.py
    """

    opened_hdf5() # To be sure as there were some weird opening errors.
    assert os.path.exists(filepath), 'file %s not found' % filepath

    with h5py.File(filepath, 'r') as h5f:
        assert key in [key for key in h5f.keys()], 'key %s does not exist in %s' % (key, filepath)
        tensor = h5f[key][()]
        #h5f.close()
        return tensor


def opened_hdf5():
    """
    Close all open HDF5 files and report number of closed files.
    :return: number of closed files
    :rtype: int

    TAKEN FROM:
      * https://github.com/davidstutz/disentangling-robustness-generalization/blob/master/common/utils.py
    """

    opened = 0
    for obj in gc.get_objects():  # Browse through ALL objects
        try:
            # is instance check may also fail!
            if isinstance(obj, h5py.File):  # Just HDF5 files
                obj.close()
                opened += 1
        except:
            pass  # Was already closed
    return opened
