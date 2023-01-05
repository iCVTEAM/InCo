import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .loader import *
import cv2 as cv
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation 
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img)) 
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


def process_dir(dir_path):
    cat_container = sorted(os.listdir(dir_path))
    cats2label = {cat:label for label, cat in enumerate(cat_container)}                                                                                                                                                          
    labels = []
    images = []
    path_imgs = []
    for cat in cat_container: 
        for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
             # if '.JPEG' not in img_path:
            # print(img_path)
            if '.jpg' not in img_path:
                continue
            label = cats2label[cat]
            path_img = os.path.join(dir_path, cat, img_path)
            #print(path_img, label)
            image =  cv.imread(path_img)
            labels.append(label)        
            images.append(image)
            path_imgs.append(path_img)
            #print(len(images))
    return images, labels, path_imgs

class ImageNet(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        self.i=[]
        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    ###default transforms 
                    
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                    
                    ###============SimClR transforms=========
                    #transforms.RandomResizedCrop(size=84),
                    #transforms.RandomCrop(84, padding=8),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),    
                    
                    ###=============MOCO_V2===========
                    #transforms.RandomCrop(84, scale=(0.2, 1.)),                    
                    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4,0.4, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),
                    #transforms.RandomApply([loader.GaussianBlur([.1,2.])], p=0.5),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ToTensor(),
                    #self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                   
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        self.transform_noaug = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize(256),
                transforms.CenterCrop(224),                   
                transforms.ToTensor(),
                self.normalize
            ])
        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'           
            #self.file_pattern = '%s'
        else:
            #self.file_pattern = '%s'
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'           
        self.data = {}
        
        #print('paty:', self.data_root)  
        ### 读取pickle格式 ### 
        #self.data_root = '/media/space/WH/WH/fewshot-CAN'
        #with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
        #    data = pickle.load(f, encoding='latin1')
        #    self.imgs = data['data']
        #    self.labels = data['labels']
        #print(self.imgs.shape[0], self.labels.shape)
        ### 读取image形式 ###
        #self.data_root = '/media/space/WH/WH/datasets/miniImagenet/'
        self.data_root = '/media/cvteam1/WH/WH/datasets/mini-imagenet-ori/images_mini_224'
        # self.data_root = '/media/cvteam1/WH/WH/datasets/image_cub_2011/images_cub/images'
        # self.data_root = '/media/cvteam1/WH/WH/datasets/images_cars/images_cars' 
        # self.data_root = '/media/cvteam1/WH/WH/datasets/dogs/images_dogs/' 
        #print(self.data_root)
        #self.data_root = '/media/space/WH/WH/datasets/images_cars'
        #self.data_root = '/media/space/WH/WH/baseline/'
        self.file_pattern = '%s'
        path_img = os.path.join(self.data_root, partition)
        self.imgs, self.labels, self.path_images = process_dir(path_img)        
        # print(len(self.imgs), len(self.labels), len(path_images))
        #print(self.train, self.train_labels2inds, self.train_labelIds)
        #print(len(self.labels),len(self.imgs)) ##12000 12000
        #print(self.labels,self.imgs) 
        # pre-process for contrastive sampling
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1
            #print(num_classes)
            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
    def __getitem__(self, item):
        # i=self.i
        #i.append(i)
        #print(len(i))
        #print(self.imgs,item)
        img = np.asarray(self.imgs[item]).astype('uint8')
        img1 = self.transform(np.array(img))
        # print('item',item)
        #img2 = self.transform(np.array(img))
        #img2 = self.transform_noaug(np.array(img))
        #img2_90 = self.transform_noaug(rotate_img(np.array(img),90))
        #img2_180 = self.transform_noaug(rotate_img(np.array(img),180))
        #img2_270 = self.transform_noaug(rotate_img(np.array(img),270))
        target = self.labels[item] - min(self.labels)
        #img2_rotate  = torch.cat((img2.unsqueeze(0), img2_90.unsqueeze(0), img2_180.unsqueeze(0), img2_270.unsqueeze(0)), 0)
        #target_rotate = torch.cat((0*torch.ones(1),torch.ones(1),2*torch.ones(1),3*torch.ones(1)), 0)
        #print(img2.shape, img2_90.shape, img2_180.shape, img2_270.shape,img2_rotate.shape, target_rotate.shape)
        if not self.is_sample:
            #print(img.shape, target)
            #print(image)
            #return img1, img2,img2_rotate, target_rotate, target, item
            return img1, img2, target, item
            #return img, img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target ,item, sample_idx
        
    def __len__(self):
        return len(self.labels)


class MetaImageNet(ImageNet):
    
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaImageNet, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        if train_transform is None:
            self.train_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(224, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(len(self.imgs)):
            #print('self', len(self.imgs), idx)
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

        #print(self.classes, self.data.keys())
    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            #print('idx:',idx,imgs.shape, cls, self.n_queries)
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            #print('supprot', support_xs_ids_sampled)
            #print('query', query_xs_ids)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))
                
        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        
        support_xs = torch.stack(list(map(lambda x: self.test_transform(np.array(x.squeeze())), np.array(support_xs))))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(np.array(x.squeeze())),np.array(query_xs))))
        # print(support_xs.shape, support_ys.shape) 
        return support_xs, support_ys, query_xs, query_ys      
        
    def __len__(self):
        return self.n_test_runs
    
    
if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = 'data'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    imagenet = ImageNet(args, 'val')
    #print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)
    
    metaimagenet = MetaImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
