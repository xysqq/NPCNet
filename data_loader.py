import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from data_transform import HorizontalFlip, VerticalFlip, Rotate, ToTensor, Resize
from glob import glob
from pathlib import Path


def get_images_labels_from_dir(img_dir, mask_dir, fold=0, mode='train'):
    images = glob(f"{img_dir}/*")

    patient_id_list = sorted(set(Path(image).stem.split('_')[0] for image in images))
    n = len(patient_id_list)

    start_index = fold * 0.2
    if start_index == 0.8:
        train_patient_id_list = patient_id_list[:int(start_index * n)]
        test_patient_id_list = patient_id_list[int(start_index * n):]
    else:
        train_patient_id_list = patient_id_list[:int(start_index * n)] + patient_id_list[
                                                                         int((start_index + 0.2) * n):]
        test_patient_id_list = patient_id_list[int(start_index * n):int((start_index + 0.2) * n)]

    choose_patient_id_list = train_patient_id_list if mode == 'train' else test_patient_id_list
    choose_images = list(sorted([f"{img_dir}/{Path(image).name}" for image in images if
                                 Path(image).stem.split('_')[0] in choose_patient_id_list]))
    choose_masks = [f"{mask_dir}/{Path(image).name}" for image in choose_images]

    return choose_images, choose_masks


class NPCDataset(Dataset):

    def __init__(self, img_dir, mask_dir, fold=0, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.labels = [0, 1, 2]
        self.images, self.masks = get_images_labels_from_dir(img_dir, mask_dir, fold, mode)
        self.hf = HorizontalFlip(p=1)
        self.vf = VerticalFlip(p=1)
        self.rt = Rotate(degrees=(90, 180, 270))
        self.rs = Resize(scales=[(320, 320), (192, 192), (384, 384), (128, 128)], p=0.5)
        self.tt = ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = np.load(self.images[item])
        mask = np.load(self.masks[item])

        if self.mode == "train":
            seed = np.random.randint(0, 4, 1)
            if seed == 0:
                pass
            elif seed == 1:
                image, mask = self.hf(image, mask)
            elif seed == 2:
                image, mask = self.vf(image, mask)
            elif seed == 3:
                image, mask = self.rt(image, mask)

            image, mask = self.tt(image, mask, labels=self.labels)

        elif self.mode == 'val':

            image, mask = self.tt(image, mask, labels=self.labels)

        elif self.mode == 'test':

            return image, mask, self.images[item]

        else:
            print("invalid transform mode")

        return image, mask


def get_dataloader(img_dir, mask_dir, fold, batch_size, num_workers, mode="train"):
    if mode == "train":
        train_dataset = NPCDataset(img_dir, mask_dir, fold, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      drop_last=True)
        return train_dataloader
    elif mode == "test":
        test_dataset = NPCDataset(img_dir, mask_dir, fold, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_dataloader
    else:
        val_dataset = NPCDataset(img_dir, mask_dir, fold, mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return val_dataloader


if __name__ == "__main__":
    image_dir = "./data/multimodal/images"
    label_dir = "./data/multimodal/labels"
    k_fold = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(image_dir, label_dir, k_fold, batch_size=1, num_workers=1, mode="train")
    val_loader = get_dataloader(image_dir, label_dir, k_fold, batch_size=1, num_workers=1, mode="val")

    train_dataset = NPCDataset(image_dir, label_dir, mode="train")
    image, mask = train_dataset[0]
    print(image.shape)
    print(mask.shape)

    # for image, mask in val_loader:
    #
    #     image = image.to(device)
    #     image = (image > 0.5).float()
    #     mask = mask.to(device)
    #     print(image.shape)
    #     print(mask.shape)
    #     print(torch.unique(mask))
    #     print(torch.unique(image))
    #
    #     # plt.imshow(image[0, 0, :, :])
    #     # plt.pause(0.1)
    #     # plt.imshow(mask[0, 0, :, :])
    #     # plt.pause(0.1)
    #
    #     break
