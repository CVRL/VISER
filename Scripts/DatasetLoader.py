import os
import sys
import torch
import random as random_lib
import torch.utils.data as data_utl
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode


class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, c2i={}, map_location='',map_size=7,im_size=224,network='densenet',keeprate=1):
        self.class_to_id = c2i
        self.id_to_class = []
        self.map_location = map_location
        self.map_size = map_size
        self.image_size = im_size
        self.keeprate = keeprate

        # Class assignment
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0

        # Image pre-processing
        self.data = []

        self.transform = transforms.Compose([
            transforms.Resize([self.image_size, self.image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
      
        self.map_transform = transforms.Compose([
            # Use BILINEAR if AREA is missing
            transforms.Resize([self.map_size, self.map_size], interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            # Adding a small blur here softens the 'spike' of a binary mask
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) 
        ])

        # Reading data from CSV file
        SegInfo=[]
        print("Reading in data for:",train_test)
      
        with open(split_file, 'r') as f:
            for l in f.readlines():
                v= l.strip().split(',')
                if train_test == v[0]:
                    image_name = v[2]

                    imagePath = image_name
                    img = Image.open(imagePath).convert('RGB')
                    tranform_img = self.transform(img)
                    img.close()

                    v2 = os.path.basename(v[2])
                    v2 = v2.split('.')
                    v2 = v2[0]+'_blended.'+v2[1]


                    if train_test == 'Train' and os.path.exists(self.map_location + v2):
                        human_map = Image.open(self.map_location + v2).convert("L")
                        transform_human_map = self.map_transform(human_map)
                        transform_human_map = transform_human_map.type(torch.float)
                        transform_human_map = torch.squeeze(transform_human_map)

                        t_min = torch.min(transform_human_map)
                        t_max = torch.max(transform_human_map)

                        transform_human_map = transform_human_map - t_min

                        denominator = t_max - t_min
                        
                        if denominator > 1e-8:
                            transform_human_map = transform_human_map / denominator
                        else:
                            transform_human_map = torch.zeros_like(transform_human_map)
                        
                        human_map.close()
                    else:
                        transform_human_map = 0
                    c = v[1]

                    randval = random_lib.random()

                    if randval < self.keeprate:
                        if self.keeprate < 1: print(f'tag:keepimage,{image_name}')
                        keep = True
                    else:
                        print(f'tag:dropimage,{image_name}')
                        keep = False
                        
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1

                    self.data.append([imagePath, self.class_to_id[c],tranform_img[0:3,:,:],transform_human_map, keep])
        print("Class assignments:",self.class_to_id)

        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls, img, hmap, keep = self.data[index]
        imageName = imagePath.split('/')[-1]

        return img, cls, imageName, hmap

    def __len__(self):
        return len(self.data)
