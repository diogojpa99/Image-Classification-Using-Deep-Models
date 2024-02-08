import os

import torchvision.transforms as transforms
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def Build_Transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        if args.nb_classes == 2:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,            
            )
                        
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
            
        return transform

    t = []
    if resize_im and args.input_size != 224:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def Build_Dataset(is_train, data_path, args):
        
    transform = Build_Transform(is_train, args)
    root = os.path.join(data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    args.nb_classes = len(dataset.classes)
    
    return dataset