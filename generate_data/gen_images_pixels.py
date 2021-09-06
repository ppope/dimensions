"""
Modified from:
  * https://github.com/huggingface/pytorch-pretrained-BigGAN#usage

"""
import os
import json
import torch
import numpy as np
from datetime import datetime
import argparse
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names,
                                       truncated_noise_sample)
from utils import write_config
from scipy.stats import special_ortho_group
from dataloader import load_data


parser = argparse.ArgumentParser()
parser.add_argument('--num_noise_per_sample', type=int, default=10,
                   help="Total number of samples to generate")
parser.add_argument('--noise_dim', type=int, default=128,
                   help="Size of latent dimension to use." \
                        "Number of free components")
parser.add_argument('--output_shape', type=str, default="3x32x32",
                   help="Output 'image' dimension.")
parser.add_argument('--save_dir', type=str, default='samples/test',
                   help="Base level save directory for files. Actual outputs in subdirectory")
parser.add_argument('--class_ind', type=int, default=0, choices=[0,1],
                   help="Which class label to assign")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dset', type=str, default='mnist',
                   help="Which dataset to add noise to")
parser.add_argument('--max_num_samples', type=int, default=-1,
                   help="Total number of samples to generate")


def main(args):

  #seed = args.seed
  #np.random.seed(seed)
  #Create out directory is it doesn't exist
  base_save_dir = args.save_dir
  noise_dim  = args.noise_dim
  output_shape_str = args.output_shape
  C,H,W = (int(x) for x in output_shape_str.split("x"))
  image_size = (C,H,W)
  output_dim = np.prod(image_size)
  cls_id = args.class_ind
  save_dir = os.path.join(base_save_dir, "pixels_{}_noise-dim={}_cls={}".format(args.dset, noise_dim, cls_id))
  print("Save directory: {}".format(save_dir))
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  else:
   if "test" not in save_dir:
     raise Exception('Output Directory {} already exists!'.format(save_dir))

  args.save_dir = save_dir
  num_noise_per_sample = args.num_noise_per_sample
  args.batch_size = num_noise_per_sample
  write_config(args)

  #Load data
  dset = load_data(args)
  num_samples = len(dset)

  start = datetime.now()
  all_inds = np.arange(output_dim)
  for i,(x,t) in enumerate(dset):
    print('Sample: {}/{}'.format(i+1, num_samples))

    ##Generate noise
    outputsl = []
    for j in range(num_noise_per_sample):
        x_ = x.clone()
        #Sample some random pixels to change, and create mask
        rand_inds = np.random.choice(all_inds, size=noise_dim, replace=False)
        mask = np.zeros(output_dim, dtype='bool')
        mask[rand_inds] = True
        mask = mask.reshape(image_size)
        mask = torch.from_numpy(mask)
        noise = np.random.uniform(0, 1, size=noise_dim).astype('float32')
        noise = torch.from_numpy(noise)
        #Replace random pixels with noise
        x_.masked_scatter_(mask, noise)
        #Add batch_dim back in
        x_ = x_.unsqueeze(0)
        outputsl.append(x_)

    #Broadcast
    outputs = torch.cat(outputsl, axis=0)
    out_fn = "{}.pt".format(i)
    out_fp = os.path.join(save_dir, out_fn)

    torch.save(outputs, out_fp)

  print('Num images generated: {}'.format(num_samples*num_noise_per_sample))
  print('Runtime: {}'.format(datetime.now() - start))


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
