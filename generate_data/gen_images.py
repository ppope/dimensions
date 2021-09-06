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

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, default='biggan-deep-128',
                   help="Pretrained GAN model: choice of 128,256,512")
parser.add_argument('--num_samples', type=int, default=1024,
                   help="Total number of samples to generate")
parser.add_argument('--latent_dim', type=int, default=128,
                   help="Size of latent dimension to use." \
                         "Decreased from original by fixing first k components to be zero")
parser.add_argument('--batch_size', type=int, default=128,
                   help="Batch size for generating images")
parser.add_argument('--class_name', type=str, default='soap bubble',
                   help="Wordnet name of Imagenet class to generate")
parser.add_argument('--class_id', type=int, default=0,
                   help="ID of class {0,...,999}")
parser.add_argument('--truncation', type=float, default=1,
                   help="Level of Truncation in sampling density" \
                        "Trades off between diversity (1) and sample quality (0)." \
                        "See the paper for details.")
parser.add_argument('--save_dir', type=str, default='samples/test',
                   help="Save directory for files")
parser.add_argument('--gpu_id', type=str, default="0",
                   help="GPU ID(s) to use")
parser.add_argument('--add', action='store_true', default=False,
                   help="Adds more data to existing directory" \
                        "Overides directory check")

NUM_IMAGENET_CLS = 1000


def main(args):
  #Create out directory is it doesn't exist
  if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
  else:
   if "test" not in args.save_dir and not args.add:
     raise Exception('Output Directory {} already exists!'.format(args.save_dir))
  print("Saving to {}".format(args.save_dir))
  write_config(args)
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

  # Load pre-trained model tokenizer (vocabulary)
  model = BigGAN.from_pretrained(args.model_file)
  if args.gpu_id:
    model.to('cuda')

  num_batches = args.num_samples // args.batch_size

  if args.add:
    existing_batches = [x for x in os.listdir(args.save_dir) if ".pt" in x]
    start_batch_num = max([ int(x.split("/")[-1].replace(".pt", "")) for x in existing_batches])
    num_batches += start_batch_num
  else:
    start_batch_num = 0

  start = datetime.now()
  for b in range(start_batch_num, num_batches):
    print('Batch: {}/{}'.format(b+1, num_batches))

    # Prepare inputs
    if args.class_id:
      class_vector = np.zeros((args.batch_size, NUM_IMAGENET_CLS), dtype=np.float32)
      class_vector[:, args.class_id] = 1
    elif args.class_name:
      class_vector = one_hot_from_names([args.class_name], batch_size=args.batch_size)
    else:
      raise Exception("Must specify either class name or ID!")
    noise_vector = truncated_noise_sample(truncation=args.truncation, batch_size=args.batch_size)
    latent_dim_orig = noise_vector.shape[1]
    if latent_dim_orig != args.latent_dim:
      #Reduce dimension of noise_vector by fixing components to be zero
      assert args.latent_dim < latent_dim_orig
      k = latent_dim_orig - args.latent_dim
      noise_vector[:,:k] = 0

    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)
    if args.gpu_id:
      noise_vector = noise_vector.to('cuda')
      class_vector = class_vector.to('cuda')

    # Generate imagex
    with torch.no_grad():
      output = model(noise_vector, class_vector, args.truncation)

    out_fn = "{}.pt".format(b)
    out_fp = os.path.join(args.save_dir, out_fn)
    if args.gpu_id:
      output = output.to('cpu')

    ##Transform images from [-1,1] to [0,1]
    output = (output + 1)*(0.5)
    torch.save(output, out_fp)

  print('Num images generated: {}'.format(args.num_samples))
  print('Runtime: {}'.format(datetime.now() - start))


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
