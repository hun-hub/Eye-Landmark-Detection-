import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
import json
from glob import glob
from dataclasses import dataclass
from typing import List
import random
@dataclass
class DataInfo:
    rgb_mean: List[float]
    rgb_stddev: List[float]
    joint_names: List[str]
    hflip_indices: List[int]
    
class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.input_res = 256
        self.output_res = 64
        self.generateHeatmap = GenerateHeatmap(self.output_res, 28)
        self.path = path
        self.img_list = glob(self.path+'/*/*.jpg')
        self.json_list = glob(self.path+'/*/*.json')
        # shuffle img_list

        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.loadImage(idx)

    def loadImage(self, idx):
        orig_img = cv2.imread(self.img_list[idx])
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

        ratio_x = self.output_res/640
        ratio_y = self.output_res/480

        orig_keypoints = json.load(open(self.json_list[idx]))
        eyelid_x = orig_keypoints['eyelid_x']
        eyelid_x = [x*ratio_x for x in eyelid_x]
        eyelid_y = orig_keypoints['eyelid_y']
        eyelid_y = [y*ratio_y for y in eyelid_y]
        if len(eyelid_x) != len(eyelid_y):
            print('eyelid_x, eyelid_y length is not same')
            print(self.json_list[idx])
        eyelid_coord = np.array([eyelid_x, eyelid_y]).T

        iris_x = orig_keypoints['iris_x']
        iris_x = [x*ratio_x for x in iris_x]
        iris_y = orig_keypoints['iris_y']
        iris_y = [y*ratio_y for y in iris_y]
        if len(iris_x) != len(iris_y):
            print('iris_x, iris_y length is not same')
        iris_coord = np.array([iris_x, iris_y]).T

        pupil_x = orig_keypoints['pupil_x']
        pupil_x = [x*ratio_x for x in pupil_x]
        pupil_y = orig_keypoints['pupil_y']
        pupil_y = [y*ratio_y for y in pupil_y]
        if len(pupil_x) != len(pupil_y):
            print('pupil_x, pupil_y length is not same')
        pupil_coord = np.array([pupil_x, pupil_y]).T

        orig_keypoints = np.concatenate((eyelid_coord, iris_coord, pupil_coord), axis=0)

        keypoints = np.copy(orig_keypoints)

        ## generate heatmaps on outres
        generateHeatmap = GenerateHeatmap(output_res=64, num_parts=28)
        # generateHeatmap_npy = generateHeatmap(keypoints.reshape(1, -1, 2))
        # # save npy
        # cv2.imwrite('./npy/test.jpg', orig_img)
        # np.save('./npy/test.npy', generateHeatmap_npy)
        
        heatmaps = generateHeatmap(keypoints.reshape(1, -1, 2))
        orig_img = cv2.resize(orig_img, (256, 256)).reshape(3, 256, 256) / 255.0
        meta= {'index' : idx}
        
        return orig_img.astype(np.float32), heatmaps.astype(np.float32), meta

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data


def init(config):
    batchsize = config['train']['batchsize']

    train_path = './data/MPII/final'
    valid_path = './data/MPII/valid'

    datasets = {
        'train': Dataset(train_path),
        'valid': Dataset(valid_path)
    }

    # 데이터 로더 생성
    loaders = {
        key: torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=config['train']['num_workers'], pin_memory=False, drop_last=True)
        for key, dataset in datasets.items()
    }

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            try:
                imgs, heatmaps = next(loader)
                # print(imgs.shape)
            except StopIteration:
                loader = loaders[phase].__iter__()
                imgs, heatmaps = next(loader)
            yield {
                'imgs': imgs,
                'heatmaps': heatmaps
            }

    return lambda key: gen(key)