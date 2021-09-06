import torchvision
from torchvision import transforms
import imagehash
import numpy as np
import time

# trainset = torchvision.datasets.ImageFolder(
#                     "/datasets/ILSVRC2012/train")

trainset = torchvision.datasets.ImageFolder(
                    "/fs/cml-datasets/ImageNet/ILSVRC2012")

hashing_list = []
last_time = time.time()
for n, (img, tgt) in enumerate(trainset):
    hash = imagehash.average_hash(img)
    hashing_list.append(str(hash))
    if n % 10000 == 0:
        print("Finished {}/{} images in {} s".format(n, len(trainset), time.time()-last_time))

_, uidx = np.unique(hashing_list, return_index=True)

print("Got {} unique images out of {} images".format(len(uidx), len(hashing_list)))
np.save("imagenet_uid_cml.npy", uidx)
