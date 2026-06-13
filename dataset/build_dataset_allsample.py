import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .cifar import cifar10, cifar100


def get_transforms(n_px=224, transform_mode='origin'):
    if transform_mode == 'origin':
        transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
    elif transform_mode == 'flip':
        transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
    return transform


class FewShotDatasetWrapper(Dataset):
    def __init__(self, db, select_labels):
        self.db = db
        self.select_labels = select_labels

    def __getitem__(self, index):
        db_idx = self.select_labels[index]
        return self.db.__getitem__(db_idx)

    def __len__(self):
        return len(self.select_labels)

    def prompts(self, mode='single'):
        return self.db.prompts(mode)

    def get_labels(self):
        return self.db.get_labels()

    def get_classes(self):
        return self.db.get_classes()


def get_split_index(labels, n_shot, n_val=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    all_label_list = np.unique(labels)
    train_idx_list = []
    val_idx_list = []
    for label in all_label_list:
        label_collection = np.where(labels == label)[0]
        random.shuffle(label_collection)
        selected_idx = label_collection[:n_shot + n_val]
        train_idx_list.extend(selected_idx[:n_shot])
        val_idx_list.extend(selected_idx[n_shot:])
    return train_idx_list, val_idx_list


def build_dataset(db_name, root, n_shot=-1, n_val=0, transform_mode='origin'):
    transform = get_transforms(transform_mode=transform_mode)
    test_transform = get_transforms(transform_mode='origin')

    db_func = {
        'cifar10': cifar10,
        'cifar100': cifar100,
    }

    train_db = db_func[db_name](root, transform=transform, train=True)
    test_db = db_func[db_name](root, transform=test_transform, train=False)

    return train_db, test_db
