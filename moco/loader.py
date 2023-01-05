# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import numpy as np
from numpy.linalg.linalg import transpose
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.transforms.transforms import CenterCrop
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation 
        #print(img.shape)
        img1 = img[0,:,:]
        img2 = img[1,:,:]
        img3 = img[2,:,:]
        img[0,:,:]=np.rot90(img1, 1)
        img[1,:,:]=np.rot90(img1, 1)
        img[2,:,:]=np.rot90(img1, 1)
        return img
        #return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        img1 = img[0,:,:]
        img2 = img[1,:,:]
        img3 = img[2,:,:]
        img[0,:,:]=np.rot90(img1, 2)
        img[1,:,:]=np.rot90(img2, 2)
        img[2,:,:]=np.rot90(img3, 2)
        return img
        #return np.fliplr(np.flipud(img)) 
    elif rot == 270: # 270 degrees rotation / or -90
        img1 = img[0,:,:]
        img2 = img[1,:,:]
        img3 = img[2,:,:]
        img[0,:,:]=np.rot90(img1, 3)
        img[1,:,:]=np.rot90(img2, 3)
        img[2,:,:]=np.rot90(img3, 3)
        return img
        #return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        augmentation = [
            #transforms.Resize(96),
            #transforms.CenterCrop(84),
            #transforms.RandomCrop(84, padding=8),
            #Image.fromarray(),
            transforms.RandomResizedCrop(84, scale=(0.2, 1.)),
            #transforms.CenterCrop(84),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            #normalize
        ]
        self.base_transform_rote = transforms.Compose(augmentation)
    def __call__(self, x):
        #x=Image.open(x)
        #to_img = transforms.ToPILImage()
        #x= to_img(x)
        #print(x) 
        img2 = torch.from_numpy(rotate_img(np.array(self.base_transform_rote(x)), 0).copy())
        img2_90 = torch.from_numpy(rotate_img(np.array(self.base_transform_rote(x)),90).copy())
        img2_180 =  torch.from_numpy(rotate_img(np.array(self.base_transform_rote(x)),180).copy())
        img2_270 =  torch.from_numpy(rotate_img(np.array(self.base_transform_rote(x)),270).copy())  
        #print(img2.shape, img2_90.shape, img2_180.shape, img2_270.shape)
        img_rotate  = torch.cat((img2.unsqueeze(0), img2_90.unsqueeze(0), img2_180.unsqueeze(0), img2_270.unsqueeze(0)), 0)
        target_rotate = torch.cat((0*torch.ones(1),torch.ones(1),2*torch.ones(1),3*torch.ones(1)), 0) 
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k, img_rotate, target_rotate]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
