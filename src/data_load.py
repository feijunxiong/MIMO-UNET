import os
from PIL import Image as Image
import mindspore.dataset as Dataset
from .data_augment import PairRandomCrop, PairRandomHorizontalFlip, PairToTensor, PairCompose, RandomRGB
import numpy as np

def build_dataset(path, mode='train'):
    if mode == 'train':
        image_dir = os.path.join(path, 'train')
        transform = PairCompose([PairRandomCrop(256), PairRandomHorizontalFlip(), RandomRGB(), PairToTensor()]) 
    elif mode == 'test' :
        image_dir = os.path.join(path, 'test')
        transform = PairCompose([PairToTensor()])
    dataset = DeblurDataset(image_dir, transform=transform)
    return dataset

def build_dataloader(dataset, column_names=None, shuffle=None, num_parallel_workers=1, batch_size=1):
    dataloader = Dataset.GeneratorDataset(
        dataset,
        column_names=column_names,
        shuffle=shuffle,
        num_parallel_workers=num_parallel_workers
    )
    dataloader = dataloader.batch(batch_size=batch_size)
    return dataloader

class DeblurDataset:
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list_blur = os.listdir(os.path.join(image_dir, 'blur/'))
        self.image_list_sharp = os.listdir(os.path.join(image_dir, 'sharp/'))
        self._check_image(self.image_list_blur)
        self._check_image(self.image_list_sharp)
        self.image_list_blur.sort()
        self.image_list_sharp.sort()
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, 'blur', self.image_list_blur[index]))
        label = Image.open(os.path.join(self.image_dir, 'sharp',self.image_list_sharp[index]))
        image = np.asarray(image)
        label = np.asarray(label)

        image, label = self.transform(image, label)

        return image, label
    
    def __len__(self):
        return len(self.image_list_blur)
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
