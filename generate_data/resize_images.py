import os
import argparse
import json
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--top_dir', default='gen_gan/samples', type=str)
parser.add_argument('--in_data_name', default='basenji_128', type=str)
parser.add_argument('--max_num_samples', default=10000, type=int)
args = parser.parse_args()

top_dir = args.top_dir
in_data_name = args.in_data_name
max_num_samples = args.max_num_samples

input_dir = os.path.join(top_dir, in_data_name)
#Load config for input data
config_fp = os.path.join(input_dir, "config.json")
with open(config_fp, 'r') as fh:
    config = json.load(fh)

batch_size = config['batch_size']
num_samples = config['num_samples']
num_batches = min(num_samples, max_num_samples) // batch_size

#Ensure your input data is of this size!!
in_data_size = (3,128,128)
print("Old image size: {}".format(in_data_size))
scale_factors = [1/8.0, 1/4.0, 1/2.0, 1, 2]

#Load, resize and save images
for scale_factor in scale_factors:

    out_data_size = (in_data_size[0],
                     int(in_data_size[1]*scale_factor),
                     int(in_data_size[2]*scale_factor))

    print("New image size: {}".format(out_data_size))

    out_data_size_str = "x".join([str(x) for x in out_data_size])

    out_data_name = "{}_edim={}".format(in_data_name, out_data_size_str)
    output_dir = os.path.join(top_dir, out_data_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #Save config to new dir
    config['old_image_size'] = in_data_size
    config['image_size'] = out_data_size
    config_fp = os.path.join(output_dir, "config.json")
    with open(config_fp, 'w') as fh:
        json.dump(config, fh)

    for b in range(num_batches):
        #Load
        fn = "{}.pt".format(b)
        in_fp = os.path.join(input_dir, fn)
        images = torch.load(in_fp)
        #Resize
        images = torch.nn.functional.interpolate(images.to(torch.float),
                                                 scale_factor=scale_factor,
                                                 mode='nearest')
        #images = images.to(torch.uint8)
        #Save
        out_fp = os.path.join(output_dir, fn)
        torch.save(images, out_fp)
