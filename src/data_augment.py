from mindspore.dataset.vision.c_transforms import HorizontalFlip
from mindspore.dataset.vision.py_transforms import ToTensor
import random
import mindspore.ops as ops
from PIL import Image
import numpy as np

class PairRandomCrop:
    def __init__(self, size=256):
        self.size = size if isinstance(size,tuple) else (size,size)

    def __call__(self, image, label):
        img_H, img_W, C = image.shape
        crop_h, crop_w = self.size

        if img_H > crop_h and img_W > crop_w:
            top = random.randint(0, img_H - crop_h)
            left = random.randint(0, img_W - crop_w)
        elif img_H == crop_h and img_W == crop_w:
            top = 0
            left = 0

        image, label = image[top:top+crop_h, left:left+crop_w], label[top:top+crop_h, left:left+crop_w]
        return image, label

class PairRandomHorizontalFlip:
    def __init__(self, factor=0.5):
        self.factor = factor
        self.flip = HorizontalFlip()

    def __call__(self, image, label):
        if random.random() < self.factor:
            return self.flip(image), self.flip(label)
        return image, label

class PairToTensor:
    def __init__(self):
        self.totensor = ToTensor()

    def __call__(self, image, label) :
        return self.totensor(image), self.totensor(label)

class PairCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class RandomRGB(object):
    """Random permute R,G,B
    """
    def __init__(self, prob=0.2):
        self.prob = prob
        self.channel_range = [
            [0,2,1],
            [1,0,2],
            [1,2,0],
            [2,0,1],
            [2,1,0]
        ]

    def __call__(self, lr, hr):
        """Random Permute the R,G,B
        Args:
            lr: pil or ndarray
            hr: pil or ndarray
        Returns:
            PIL image.
        """
        if random.random() < self.prob:
            random_permute = random.choice(self.channel_range)

            if isinstance(lr, Image.Image):
                lr = np.array(lr)
                hr = np.array(hr)
            elif isinstance(lr, np.ndarray):
                lr = lr
                hr = hr
            else:
                raise TypeError("input must be PIL or np.ndarray format!!!")

            lr = lr[:, :, random_permute]
            hr = hr[:, :, random_permute]

            lr = Image.fromarray(lr)
            hr = Image.fromarray(hr)

        return lr, hr

def gse_in(image):
    flip = ops.ReverseV2(axis=[-1])
    inp = []
    
    inp.append( image ) 
    inp.append( flip(image) ) 
    return inp

def gse_pr(pre):
    flip = ops.ReverseV2(axis=[-1])
    result = []
    
    result.append( pre[0] ) 
    result.append( flip(pre[1]) ) 
    return sum(result)/2