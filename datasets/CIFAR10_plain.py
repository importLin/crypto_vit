import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image


def extract_cifar10(data_dir, save_dir, train):
    os.makedirs(save_dir, exist_ok=True)
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    if train:
        dataset = CIFAR10(data_dir, train=True, download=True, transform=transform)
    else:
        dataset = CIFAR10(data_dir, train=False, download=True, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    for i, batch_data in enumerate(data_loader):
        print(i, len(data_loader))
        img, label = batch_data
        img_id = f'{i:05}'
        label = label.item()
        save_image(img, os.path.join(save_dir, f'{img_id}_{label}.bmp'))


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


def main():
    # train_save_dir = ('./downloaded_dataset/CIFAR10_local_bmp/train')
    # extract_cifar10('./downloaded_dataset/CIFAR10_zip', train_save_dir, train=True)

    test_save_dir = './downloaded_dataset/CIFAR10_plain_bmp/test'
    extract_cifar10('./downloaded_dataset/CIFAR10_zip', test_save_dir, train=False)
    # transform = T.ToTensor()
    # local_dataset = LocalCIFAR10('img_samples', transform)
    # img, label = local_dataset.__getitem__(0)
    # print(img.shape, labl.item())


if __name__ == '__main__':
    main()