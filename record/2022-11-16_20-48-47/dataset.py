import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation.augmentation import Augmentation


def collect_pics(patients):
    patients_pics = []
    for patient in patients:
        pics = glob.glob(patient + '/*.png')
        patients_pics += pics
    return patients_pics


# 数据集类
class DataSet(Dataset):

    def __init__(self, cfg, mode):
        # 类变量定义
        self.cfg = cfg
        self.mode = mode
        self.aug = Augmentation(self.cfg)
        self.samples = []
        # 找出所有病人文件夹，共87位病例，按人划分为训练验证和测试
        patient_list = glob.glob(cfg['data_path'] + '/tumor_mask/*')
        patient_num = len(patient_list)
        train_num = int(patient_num * cfg['train_ratio'])
        test_num = patient_num - train_num
        random.shuffle(patient_list)
        train_patients = patient_list[:train_num]
        test_patients = patient_list[-test_num:]
        self.train_samples = collect_pics(train_patients)
        self.test_samples = collect_pics(test_patients)

    def __getitem__(self, idx):
        if self.mode == 'train':
            name = self.train_samples[idx]
        else:
            name = self.test_samples[idx]
        image = cv2.imread(name.replace('tumor_mask',
                                        'train_img'), 0).astype(np.float32)
        mask = cv2.imread(name, 0).astype(np.float32)
        mask /= 255
        # resnet只能接受三通道输入
        image = np.dstack((image, image, image))

        if self.mode == 'train':
            image, mask = self.aug.normalize(image, mask)
            image, mask = self.aug.random_crop(image, mask)
            image, mask = self.aug.random_flip(image, mask)
            return image, mask, name
        else:
            image, mask = self.aug.normalize(image, mask)
            return image, mask, name

    def collate(self, batch):
        size = 512
        image, mask, name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i],
                                  dsize=(size, size),
                                  interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i],
                                 dsize=(size, size),
                                 interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask, name

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_samples)
        else:
            return len(self.test_samples)
