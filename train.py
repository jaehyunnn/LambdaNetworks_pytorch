from time import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argparse import ArgumentParser

from models.lambda_resnet import *

from torch.nn.parallel import DataParallel
from torch.nn import Parameter

import warnings
warnings.filterwarnings("ignore")

def train(**args):
    gpu = list(map(int, args['gpu']))
    print("* Using GPU - %s\n"%(str(gpu)))

    # Model
    model = lambda_resnet50(num_classes=10)

    # Set device & Data Parallelization
    torch.cuda.set_device(gpu[0])
    if len(gpu) > 1:  # Using multiple-GPUs
        model = DataParallel(model, device_ids=gpu)
    model.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset_valid = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=args['batch_size'], num_workers=len(gpu)*4)
    dataloader_valid = DataLoader(dataset_valid, shuffle=False, batch_size=args['batch_size'], num_workers=len(gpu)*4)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

    for epoch in range(args['num_epochs']):
        """ Training iteration """
        model.train()
        for batch_idx, (samples, labels) in enumerate(dataloader_train):
            optimizer.zero_grad()

            if gpu:
                samples = samples.cuda()
                labels = labels.cuda()

            logits = model(samples)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(samples), len(dataloader_train.dataset),
                           100. * batch_idx / len(dataloader_train), loss.item()))
        scheduler.step()

        """ Validation iteration """
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            correct = 0
            for samples, labels in dataloader_valid:
                if gpu:
                    samples = samples.cuda()
                    labels = labels.cuda()

                logits = model(samples)
                valid_loss += criterion(logits, labels)
                preds = logits.argmax(dim=1, keepdim=True)
                correct += preds.eq(labels.view_as(preds)).sum().item()
            valid_loss /= len(dataloader_valid)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                valid_loss, correct, len(dataloader_valid.dataset), 100. * correct / len(dataloader_valid.dataset)))


if __name__ == '__main__':
    print()
    print("* Python version\t: ", sys.version)
    print("* PyTorch version\t: ", torch.__version__)
    print("* CUDA version\t\t: ", torch.version.cuda)

    print(r"  _____ ____      _    ___ _   _ ___ _   _  ____", "\n",
          r"|_   _|  _ \    / \  |_ _| \ | |_ _| \ | |/ ___|", "\n",
          r"  | | | |_) |  / _ \  | ||  \| || ||  \| | |  _", "\n",
          r"  | | |  _ <  / ___ \ | || |\  || || |\  | |_| |", "\n",
          r"  |_| |_| \_\/_/   \_\___|_| \_|___|_| \_|\____|", "\n")

    parser = ArgumentParser()

    parser.add_argument("--gpu", type=list, action="append", default=[2], help="GPU IDs")

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num-epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--log-interval', type=int, default=100, help='training log interval')


    train(**vars(parser.parse_args()))