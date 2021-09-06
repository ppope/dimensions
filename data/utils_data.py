import torch
import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings
from torchvision import transforms

from torchvision.transforms import functional as F

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class PaddedCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

        crop_pos: The coordinate of the the top-left corner to put the cropping box on the padded image.
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant', crop_pos=(0, 0)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.crop_pos = crop_pos

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = self.crop_pos[0]
        j = self.crop_pos[1]
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1}, pos={2})'.format(self.size, self.padding, self.crop_pos)

class HorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        return F.hflip(img)

    def __repr__(self):
        return self.__class__.__name__

def identity():
    return lambda x: x

def hflip():
    return HorizontalFlip()


def c_0_1(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(0, 1))

def c_0_2(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(0, 2))

def c_0_3(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(0, 3))

def c_1_0(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(1, 0))

def c_1_1(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(1, 1))

def c_1_2(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(1, 2))

def c_1_3(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(1, 3))

def c_2_0(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(2, 0))

def c_2_1(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(2, 1))

def c_2_2(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(2, 2))

def c_2_3(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(2, 3))

def c_3_0(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(3, 0))

def c_3_1(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(3, 1))

def c_3_2(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(3, 2))

def c_3_3(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
    return PaddedCrop(size, padding, pad_if_needed, fill, padding_mode, crop_pos=(3, 3))


def get_multiaug_cifar(dataset, base_tfm, add_tfms):
    """

    :param dataset:
    :param base_tfm:
    :param add_tfms: a list of transformation names
    :return:
    """
    # trainset = dataset(root='./data',
    #                 train=True,
    #                 download=True,
    #                 transform=transforms.Compose(base_tfm), unique=True)
    # train_datasets = [trainset]
    # Update: now need to specify the identity transform if needed
    img_list, target_list = [], []
    for tfm_name in add_tfms:
        if tfm_name in ["hflip", "identity"]:
            tfm = eval(tfm_name)()
        elif "contrast" in tfm_name:
            con = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(contrast=con)
        elif "brightness" in tfm_name:
            bri = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(brightness=bri)
        elif "saturation" in tfm_name:
            sat = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(saturation=sat)
        elif "hue" in tfm_name:
            hue = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(hue=hue)
        elif tfm_name == "rand_hflip":
            tfm = transforms.RandomHorizontalFlip()
        else:
            tfm = eval(tfm_name)(32, padding=4)

        dset = dataset(root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([tfm] + base_tfm), unique=True)
        # apply the new transform once to the dataset
        for img, label in dset:
            img_list.append(img)
            target_list.append(label)
    # get the unique images
    imgs, targets = np.stack(img_list), np.stack(target_list)
    imgs, uidx = np.unique(imgs, return_index=True, axis=0)
    targets = targets[uidx]

    return torch.utils.data.TensorDataset(torch.Tensor(imgs), torch.Tensor(targets))


def get_single_img_multiaug_cifar(dataset, base_tfm, add_tfms, img_idx, sample_times):
    """

    :param dataset:
    :param base_tfm:
    :param add_tfms: a list of transformation names
    :return:
    """
    # trainset = dataset(root='./data',
    #                 train=True,
    #                 download=True,
    #                 transform=transforms.Compose(base_tfm), unique=True)
    # train_datasets = [trainset]
    # Update: now need to specify the identity transform if needed
    img_list, target_list = [], []
    tfm_list = []
    for tfm_name in add_tfms:
        if tfm_name in ["hflip", "identity"]:
            tfm = eval(tfm_name)()
        elif "contrast" in tfm_name:
            con = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(contrast=con)
        elif "brightness" in tfm_name:
            bri = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(brightness=bri)
        elif "saturation" in tfm_name:
            sat = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(saturation=sat)
        elif "hue" in tfm_name:
            hue = float(tfm_name.split('_')[1])
            tfm = transforms.ColorJitter(hue=hue)
        elif tfm_name == "rand_hflip":
            tfm = transforms.RandomHorizontalFlip()
        else:
            tfm = eval(tfm_name)(32, padding=4)

        tfm_list.append(tfm)

    dset = dataset(root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(tfm_list + base_tfm), unique=True)
    # apply the new transform once to the dataset
    for s in range(sample_times):
        img, label = dset[img_idx]
        img_list.append(img)
        target_list.append(label)
    # get the unique images
    imgs, targets = np.stack(img_list), np.stack(target_list)
    imgs, uidx = np.unique(imgs, return_index=True, axis=0)
    targets = targets[uidx]

    print("Got {} unique images.".format(imgs.shape[0]))

    return torch.utils.data.TensorDataset(torch.Tensor(imgs), torch.Tensor(targets))
