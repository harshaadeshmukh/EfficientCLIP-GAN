"""
     Files containing the functions for preparation of the training
"""

import os, sys
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import clip
import importlib
from lib.utils import choose_model

def load_clip(clip_info, device):
    """
    Import and load the CLIP model
    """
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model

def prepare_models(args):
    """
    Prepares the required model
    """
    # Set the devices, GPUs and their local ranking for distributed training from the CMD arguments
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus

    # Create the CLIP model for training and evaluation
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()

    # Create the Generator, Discriminator, Comparator model and text & image encoders
    NetG, NetD, NetC, CLIP_IMG_ENCODER, CLIP_TXT_ENCODER = choose_model(args.model)

    # Freezing the CLIP image encoders weights and set to the evaluation mode
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()

    # Freezing the CLIP text encoders weights and set to the evaluation mode
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()

    # Initializing and configuring the CLIP-GAN model
    netG = NetG(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size, args.mixed_precision, CLIP4trn).to(device)
    netD = NetD(args.nf, args.imsize, args.ch_size, args.mixed_precision).to(device)
    netC = NetC(args.nf, args.cond_dim, args.mixed_precision).to(device)

    # If the multiple GPUs are available and training is True, will move the model to the Distributed training
    # environment by wrapping into DistributedDataParallel() provided through torchrun
    if (args.multi_gpus) and (args.train):
        # Printing number of GPUs available
        print("Let's use ", torch.cuda.device_count(), " GPUs!")

        # Wrap Generator model in DistributedDataParallel() for distributed and parallel training with torchrun
        netG = torch.nn.parallel.DistributedDataParallel(
            netG,
            broadcast_buffers=False,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

        # Wrap Discriminator model in DistributedDataParallel() for distributed and parallel training with torchrun
        netD = torch.nn.parallel.DistributedDataParallel(
            netD,
            broadcast_buffers=False,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

        # Wrap Comparator model in DistributedDataParallel() for distributed and parallel training with torchrun
        netC = torch.nn.parallel.DistributedDataParallel(
            netC,
            broadcast_buffers=False,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    # returns the configured and initialize model
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, netG, netD, netC


def prepare_dataset(args, split, transform):
    # setting the input image size to 256 if its not RGB else to the given value
    if args.ch_size!=3:
        imsize = 256
    else:
        imsize = args.imsize

    # defining the transformations for the input image
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
        ])

    # import the custom Dataset class and initialize it
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    """
        Split the dataset and create the dataset object
    """
    # train dataset
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    # test dataset
    val_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    """
     Create the dataloaders to retrieve the image dataset from the folders using ImageLoader
    """
    # Defining the hyperparameters for loading the dataset such as batch size, workers
    batch_size = args.batch_size
    num_workers = args.num_workers

    # call the prepare datasets function to split into train and test datasets
    train_dataset, valid_dataset = prepare_datasets(args, transform)

    # creating the train dataloader and wrapping it for distributed training on multiple GPUs
    if args.multi_gpus==True:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            sampler=train_sampler
        )
    else:
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            shuffle='True'
        )

    # creating the validation dataloader and wrapping it for distributed training on multiple GPUs
    if args.multi_gpus==True:
        valid_sampler = DistributedSampler(valid_dataset)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            sampler=valid_sampler
        )
    else:
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            shuffle='True'
        )

    return train_dataloader, valid_dataloader, train_dataset, valid_dataset, train_sampler
