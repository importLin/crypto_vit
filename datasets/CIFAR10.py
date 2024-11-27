import os

import torch
from PIL import Image
from datasets.Handler import Data_handler
from torch.utils.data import random_split, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def split_train_val(train_set:Dataset, val_set_size:float):
    imgs_total_num = train_set.__len__()
    val_imgs_num = int(imgs_total_num * val_set_size)
    return random_split(train_set, [imgs_total_num - val_imgs_num, val_imgs_num])


class CIFAR10_handler(Data_handler):
    def __init__(self, dataset_dir, val_set_size, batch_size, num_workers):
        train_set = CIFAR10(dataset_dir, train=True, download=True, transform=train_transforms)
        if val_set_size > 0:
            train_set, val_set = split_train_val(train_set, val_set_size)
            val_set.transform = test_transforms
        else:
            print("The val set is set to be same as the testing set.")
            val_set = CIFAR10(dataset_dir, train=False, download=True, transform=test_transforms)

        test_set = CIFAR10(dataset_dir, train=False, download=True, transform=test_transforms)

        super(CIFAR10_handler, self).__init__(train_set, test_set, val_set, batch_size, num_workers)


class LocalCIFAR10(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.data_dir = data_dir
        self.img_names = os.listdir(data_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_names[idx])
        img = Image.open(img_path)
        img = self.transform(img)

        label = self.img_names[idx][:-4]
        label = int(label.split('_')[1])
        label = torch.tensor(label, dtype=torch.int8)
        return img, label

    def __len__(self):
        return len(self.img_names)


class LocalCIFAR10_handler(Data_handler):
    def __init__(self, dataset_dir, val_set_size, batch_size, num_workers):
        # train_set = LocalCIFAR10(f'{dataset_dir}/train', transform=train_transforms)
        train_set = None

        if val_set_size > 0:
            train_set, val_set = split_train_val(train_set, val_set_size)
            val_set.transform = test_transforms
        else:
            print("The val set is set to be same as the testing set.")
            val_set = LocalCIFAR10(f'{dataset_dir}/test', transform=test_transforms)

        test_set = LocalCIFAR10(f'{dataset_dir}/test', transform=test_transforms)

        super(LocalCIFAR10_handler, self).__init__(train_set, test_set, val_set, batch_size, num_workers)



def main():
    pass
    # mixup_args = {
    #     'mixup_alpha': 1.,
    #     'cutmix_alpha': 0.,
    #     'cutmix_minmax': None,
    #     'prob': 1.0,
    #     'switch_prob': 0.,
    #     'mode': 'batch',
    #     'label_smoothing': 0.1,
    #     'num_classes': 10}
    # Mixup(mixup_args)



