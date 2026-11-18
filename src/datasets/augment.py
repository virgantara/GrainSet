import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch


def get_augmentor():
    # imgaug is an optional dependency. ResNet50 training uses get_transforms(),
    # so do not require imgaug just to import this module.
    try:
        import imgaug.augmenters as iaa
    except ImportError as exc:
        raise ImportError(
            "get_augmentor() requires imgaug. Install it separately if needed."
        ) from exc

    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5),
        iaa.GaussianBlur(sigma=(0, 3.0))
    ])
    def augment(images):
        return seq.augment(images.transpose(0, 2, 3, 1)).transpose(0, 2, 3, 1)
    return augment
    
    
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, -alpha)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def mixmatch(x, y, u, model, augment_fn, T=0.5, K=2, alpha=0.75):
    xb = augment_fn(x)
    ub = [augment_fn(u) for _ in range(K)]
    qb = sharpen(sum(map(lambda i: model(i), ub)) / K)
    Ux = np.concatenate(ub, axis=0)
    Uy = np.concatenate([qb for _ in range(K)], axis=0)
    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = np.concatenate([Ux, xb], axis=0)[indices]
    Wy = np.concatenate([qb, y], axis=0)[indices]
    X, p = mixup(xb, Wx[:len(xb)], y, Wy[:len(xb)], alpha)
    U, q = mixup(Ux, Wx[len(xb):], Uy, Wy[len(xb):], alpha)
    return X, p, U, q
 

def get_transforms(RandBright_limit=0.15, RandBright_ratio=0.3,
                   RandContra_limit=0.15, RandContra_ratio=0.3):
    # Input images are read with cv2.imread() in BGR order.
    means_rgb = (0.308562, 0.251994, 0.187898)
    stds_rgb = (0.240441, 0.197289, 0.149387)
    means_bgr = means_rgb[::-1]
    stds_bgr = stds_rgb[::-1]

    train_transform = A.Compose([
        A.OneOf([
            A.Resize(256, 256),
            A.Resize(224, 224),
            A.Resize(256, 224),
            A.Resize(224, 256),
        ], p=1),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        # Equivalent to legacy RandomBrightness (no contrast change).
        A.RandomBrightnessContrast(
            brightness_limit=RandBright_limit,
            contrast_limit=0,
            p=RandBright_ratio,
        ),
        # Equivalent to legacy RandomContrast (no brightness change).
        A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=RandContra_limit,
            p=RandContra_ratio,
        ),
        A.RGBShift(20, 20, 20, p=0.3),
        A.OneOf([
            A.MedianBlur(blur_limit=5, p=0.3),
            A.Blur(blur_limit=5, p=0.3),
            A.GaussianBlur(),
        ], p=0.3),
        A.Normalize(mean=means_bgr, std=stds_bgr),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=means_bgr, std=stds_bgr),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
