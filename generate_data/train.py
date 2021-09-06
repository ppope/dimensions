'''Train GAN models with PyTorch.
   Modifed from:
    * https://github.com/zhuchen03/intrinsic-dimensions/blob/master/pretrain-cifar10.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import json
import numpy as np
import argparse

from data import FileDataset
from dataloader import load_data
import sys; sys.path.append(os.path.realpath('.'))
from models import *

NUM_CLS = 2

parser = argparse.ArgumentParser(description='PyTorch gen_gan Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--net', default="ResNet18", type=str)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--depth', default=4, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--dset', default='', type=str)
parser.add_argument('--dset_cls_0', default=None, type=str)
parser.add_argument('--dset_cls_1', default=None, type=str)
parser.add_argument('--num_train_per_cls', default=-1, type=int)
parser.add_argument('--num_test_per_cls', default=900, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_num_epochs', default=10, type=int)
parser.add_argument('--max_train_acc', default=-1, type=int)
parser.add_argument('--results_out_dir', default='gen_gan/results_train', type=str)
parser.add_argument('--model_out_dir', default='gen_gan/models', type=str)
parser.add_argument('--imagenet-dir', default="/scratch1/shared/datasets/ILSVRC2012", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--tag', default='', type=str, help="Tag for the experiment. Useful for metadata")
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_acc = 0  # best test accuracy
best_train_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Load
num_train_per_cls = args.num_train_per_cls
num_test_per_cls = args.num_test_per_cls
num_test = NUM_CLS*num_test_per_cls
num_train = NUM_CLS*num_train_per_cls


if not args.dset:
  is_synthetic = True
else:
  is_synthetic = False

if is_synthetic:
    num_per_cls = num_train_per_cls + num_test_per_cls
    dset_0 = FileDataset(args.dset_cls_0, num_per_cls, label=0)
    dset_1 = FileDataset(args.dset_cls_1, num_per_cls, label=1)
    dset = torch.utils.data.ConcatDataset([dset_0, dset_1])
    #Define train/test splits
    num_samples = NUM_CLS*num_per_cls
    all_inds = np.arange(num_samples)
    train_inds = np.random.choice(all_inds, size=num_train, replace=False)
    test_inds = np.array([x for x in all_inds if not x in train_inds])
    #Concat and partition
    trainset = torch.utils.data.Subset(dset, train_inds)
    testset = torch.utils.data.Subset(dset, test_inds)
else:
    def choose_rand_cls(dset):
        if dset == 'mnist':
            total_num_cls = 10
        elif dset == 'cifar10':
            total_num_cls = 10
        elif dset == 'cifar100':
            total_num_cls = 100
        elif dset == 'svhn':
            total_num_cls = 10
        elif dset == 'imagenet':
            total_num_cls = 1000
        elif "fonts" in dset:
            total_num_cls = 10
        else:
            raise Exception("Dataset not understood")
        all_cls_inds = np.arange(0, total_num_cls)
        cls_0, cls_1 = np.random.choice(all_cls_inds, size=NUM_CLS, replace=False)
        cls_0, cls_1 = int(cls_0), int(cls_1)
        if cls_0 > cls_1:
            cls_0, cls_1 = cls_1, cls_0
        return cls_0, cls_1

    def load_(args, cls_0, cls_1, train):
        args.class_ind = cls_0
        dset_0 = load_data(args, train=train)
        args.class_ind = cls_1
        dset_1 = load_data(args, train=train)
        dset = torch.utils.data.ConcatDataset([dset_0, dset_1])
        return dset

    if not args.dset_cls_0 and not args.dset_cls_1:
        print("Class 0/1 not given in runtime args. Selecting random classes")
        cls_0, cls_1 = choose_rand_cls(args.dset)
        args.dset_cls_0 = cls_0
        args.dset_cls_1 = cls_1
    else:
        cls_0 = args.dset_cls_0
        cls_1 = args.dset_cls_1
    print("Selected classes: {}, {}".format(cls_0, cls_1))
    args.max_num_samples = num_train_per_cls
    trainset = load_(args, cls_0, cls_1, train=True)
    args.max_num_samples = num_test_per_cls
    testset = load_(args, cls_0, cls_1, train=False)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if "ResNet" in args.net:
    net = eval(args.net)(width=args.width, ncls=NUM_CLS)
elif "VGG" in args.net:
    net = VGG(args.net, ncls=NUM_CLS)
elif args.net == "EqualWidthMLP":
    in_dim = np.prod(trainset[0][0].shape)
    net = eval(args.net)(width=args.width, depth=args.depth, in_dim=in_dim, num_classes=NUM_CLS)
else:
    net = eval(args.net)(ncls=NUM_CLS)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4
                      )

def train(epoch):
    global best_train_acc
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.view(-1).long()
        if not is_synthetic:
            targets[targets == cls_0] = 0
            targets[targets == cls_1] = 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #if batch_idx % 50 or batch_idx == len(trainloader) - 1:
        #    print(epoch, batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_train_acc:
        best_train_acc = acc
    # print("epoch {}/{}".format(epoch, args.max_num_epochs - 1), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #       % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return { "loss": train_loss / (batch_idx+1), "acc": 100*correct/total }


def test(epoch):
    global best_val_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.view(-1).long()
            if not is_synthetic:
                targets[targets == cls_0] = 0
                targets[targets == cls_1] = 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print("epoch {}/{}".format(epoch, args.max_num_epochs-1), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_val_acc:
        best_val_acc = acc
    return { "loss": test_loss / (batch_idx+1), "acc": acc }


def save_checkpoint(net, val_acc, train_acc, epoch, exp_label, out_dir):
   print('Saving..')
   state = {
       'net': net.state_dict(),
       'val_acc': val_acc,
       'train_acc': train_acc,
       'epoch': epoch,
   }
   out_fn = "{}.pt".format(exp_label)
   out_fp = os.path.join(out_dir, out_fn)
   torch.save(state, out_fp)


#main
save_dict = vars(args)
save_dict['num_train'] = num_train
save_dict['num_test'] = num_test
save_dict['num_cls'] = NUM_CLS
save_dict['seed'] = seed
save_dict['train'] = {}
save_dict['test'] = {}
save_dict['tag'] = args.tag
max_train_acc = args.max_train_acc
max_num_epochs = args.max_num_epochs
if is_synthetic:
    cls_0_v_1= "{}_vs_{}".format(args.dset_cls_0.split("/")[-1],
                             args.dset_cls_1.split("/")[-1])
else:
    cls_0_v_1= "{}_{}_vs_{}".format(args.dset, args.dset_cls_0, args.dset_cls_1)

if 'ResNet' in args.net:
    arch= "{}-w{}".format(args.net, args.width)
else:
    arch = args.net
exp_label = '{}-{}-num-train={}-num-test={}-max-epochs={}-max-train-acc={}-seed={}'.format(arch, cls_0_v_1, num_train, num_test, max_num_epochs, max_train_acc, seed)
print(exp_label)
print("Test metrics:")
converged = False
for epoch in range(start_epoch, start_epoch+max_num_epochs):
    if epoch in [int(max_num_epochs*0.5), int(max_num_epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    save_dict['train'][epoch] = train(epoch)
    save_dict['test'][epoch]  = test(epoch)
    if max_train_acc != -1 and best_train_acc >= max_train_acc:
        print("train_acc={} > {}... Stopping...".format(best_train_acc, max_train_acc))
        converged = True
        break

save_dict['converged'] = converged

#Save model
model_out_dir = args.model_out_dir
save_checkpoint(net, best_val_acc, best_train_acc, epoch, exp_label, model_out_dir)

#save results
out_fn = '{}_metrics.json'.format(exp_label)
out_fp = os.path.join(args.results_out_dir, out_fn)
with open(out_fp, 'w') as fh:
    json.dump(save_dict, fh)
