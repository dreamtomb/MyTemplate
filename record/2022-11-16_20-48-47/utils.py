import glob
import logging

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from augmentation.augmentation import Augmentation


def get_config(config_file_path='./config/config.yaml'):
    """
    该函数实现了从配置文件中读取配置参数的操作

    Args:
        config_file_path (str):配置文件(.yaml)的路径。 默认为'./config/config.yaml'。

    Returns:
        config (dict): 配置信息,调用方式为:config['lr']
    """
    file = open(config_file_path, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    config = yaml.load(file_data, Loader=yaml.FullLoader)
    return config


def statistic_mean_std(data_path='../tumor_seg/train_img/*.png'):
    """
    该函数用于统计数据集中所有训练图像(单通道)的均值和方差，用于进行图像预处理中的归一化

    Args:
        data_path (str): 数据集中所有图像的路径. 默认为'../tumor_seg/train_img/*.png'.

    Returns:
        mean (torch.tensor): 单通道图像的均值
        std (torch.tensor): 单通道图像的标准差
    """
    pic_list = glob.glob(data_path)
    # 统计所有图像的mean和std
    pic_sum = 0
    pic_sqrd_sum = 0
    pic_num = 0
    for item in tqdm(pic_list):
        pic = cv2.imread(item, 0).astype(np.float32)
        pic = torch.from_numpy(pic)
        pic_sum += torch.mean(pic, dim=[0, 1])
        pic_sqrd_sum += torch.mean(pic**2, dim=[0, 1])
        pic_num += 1
        print('{0}/{1}'.format(pic_num, len(pic_list)))
    mean = pic_sum / pic_num
    std = (pic_sqrd_sum / pic_num - mean**2)**0.5
    return mean, std


# 日志函数
def get_logger(filename, verbosity=1, name=None):
    """
    日志函数,用于将输出信息记录到日志文件中.

    Args:
        filename (str): 日志文件路径,一般为log/时间/log.txt
        verbosity (int): 过滤掉该等级以下的信息. 默认为1.
        name (str): 可以是所要记录日志的模块名字. 默认是None.

    Returns:
        logger : 直接用于记录信息logger.info等
    """
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL
    }
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    # 过滤掉debug信息
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def show_image(train_img, pred, mask, cfg, name):
    """
    画出这一张图像,并将pred_mask和gt_mask覆在其上

    Args:
        train_img (np.ndarray[H,W,3]): 训练图像
        pred (np.ndarray[H,W]): 预测mask,二值图
        mask (np.ndarray[H,W]): 真实mask,二值图
        cfg (dict): config信息
        now (datetime): datetime获取的当前时间
        name (str): 图像的名字
    """
    pred_mask = pred * 255
    gt_mask = mask * 255
    Aug = Augmentation(cfg)
    train_img, gt_mask = Aug.anti_normalize(train_img, gt_mask)
    train_mask = np.dstack((pred_mask, np.zeros([512, 512],
                                                dtype=np.uint8), gt_mask))
    gt_res = cv2.addWeighted(train_img,
                             0.6,
                             train_mask,
                             0.4,
                             0.9,
                             dtype=cv2.CV_32FC3)
    cv2.imwrite(
        '{}/{}/{}'.format(cfg['image_res_path'], cfg['now'],
                          name.split("/")[-1]), gt_res)


def show_batch_image(train_img, pred, mask, cfg, name):
    """
    画出这一个batch的所有图像,并将pred_mask和gt_mask覆在其上

    Args:
        train_img (torch.tensor[bs,3,H,W]): 训练图像
        pred (torch.tensor[bs,1,H,W]): 预测mask,二值图
        mask (torch.tensor[bs,1,H,W]): 真实mask,二值图
        now (datetime): datetime获取的当前时间
        name (List[bs]): 一批图像的名字
    """
    train_img = train_img.permute(0, 2, 3, 1).detach().cpu().numpy()
    pred = pred.detach().cpu().numpy().squeeze(1)
    mask = mask.detach().cpu().numpy().squeeze(1)
    for i in range(train_img.shape[0]):
        show_image(train_img[i], pred[i], mask[i], cfg, name[i])


def binarize(pred_mask, threshold=0.5):
    """
    将预测结果按照阈值进行二值化,大于阈值的为1,小于阈值的为0

    Args:
        pred_mask (torch.tensor): 网络的预测结果进行sigmoid的输出,代表每个像素是前景的概率值

    Returns:
        pred_mask (torch.tensor): 二值化的预测结果
    """
    pred_mask = (pred_mask >= threshold) + 0
    return pred_mask
