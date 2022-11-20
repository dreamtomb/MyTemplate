import cv2
import numpy as np
import torch


class Augmentation(object):
    def __init__(self, cfg):
        self.mean = np.array([[[cfg["mean"]["R"], cfg["mean"]["G"], cfg["mean"]["B"]]]])
        self.std = np.array([[[cfg["std"]["R"], cfg["std"]["G"], cfg["std"]["B"]]]])
        self.H = cfg["size"]
        self.W = cfg["size"]

    def normalize(self, image, mask):
        """
        实现归一化操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        image = (image - self.mean) / self.std
        return image, mask

    def anti_normalize(self, image, mask):
        """
        实现反归一化操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        image = image * self.std + self.mean
        return image, mask

    def random_crop(self, image, mask):
        """
        实现随机裁剪操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]

    def random_flip(self, image, mask):
        """
        实现随机翻转操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        rand_num = np.random.randint(3)
        if rand_num == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        elif rand_num == 1:
            return image[::-1, :, :], mask[::-1, :]
        else:
            return image, mask

    def resize(self, image, mask):
        """
        实现resize操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        image = cv2.resize(
            image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

    def to_tensor(self, image, mask):
        """
        实现numpy变成tensor操作

        Args:
            image (numpy.ndarray): 输入的单通道图像
            mask (numpy.ndarray): 输入的单通道mask

        Returns:
            image (numpy.ndarray): 归一化之后的图像
            mask (numpy.ndarray): 不作处理的mask
        """
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask
