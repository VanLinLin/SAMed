import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from pycocotools.coco import COCO
from pathlib import Path
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from PIL import Image
from torchvision import transforms


class Gastrointestinal(Dataset):
    def __init__(self, root_path, anno_path, split='train', transform=None):
        self.split = split
        self.transform = transform  # using transform in torch!
        self.root_path = Path(root_path)
        self.anno_path = Path(anno_path)
        self.coco = COCO(anno_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.coco_classes = dict([(v["id"], v["name"])
                                 for k, v in self.coco.cats.items()])

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(
            self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.root_path / image_info['file_name']

        image = np.array(Image.open(str(image_path)))

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        bboxes = []
        masks = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)

        sample = {'image': image, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = image_info['file_name'][:-4]
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # plt.title('In random rot flip')
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(label)
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # plt.title('In random rotate')
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(label)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res, split='train'):
        self.output_size = output_size
        self.low_res = low_res
        self.split = split

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.split == 'train':
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)

        x, y, c = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # why not 3?
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # plt.title('In RandomGenerator, after zoom')
            # plt.subplot(121)
            # plt.imshow(image)
            # plt.subplot(122)
            # plt.imshow(label)
        label_h, label_w = label.shape
        low_res_label = zoom(
            label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        # plt.title('low res label')
        # plt.imshow(low_res_label)
        # image = image / 255
        # image = torch.from_numpy(image.astype(np.float32))
        image = transforms.ToTensor()(image)
        # print(f'{image.unique()=}')
        # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        # print(f'{label.unique()=}')
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        # print(f'{low_res_label.unique()=}')
        sample = {'image': image, 'label': label.long(
        ), 'low_res_label': low_res_label.long()}
        return sample
