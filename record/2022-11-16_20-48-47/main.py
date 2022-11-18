import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import DataSet
from record.snapshot import snapshot
from trainer.test import test
from trainer.train import train
from utils.utils import get_config


def main():
    # 设置随机种子并指定训练显卡
    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 获取所有设置
    config_path = './config/config.yaml'
    config = get_config(config_path)
    now = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    config['now'] = now
    # 设置数据集
    train_set = DataSet(config, 'train')
    train_loader = DataLoader(train_set,
                              collate_fn=train_set.collate,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle']['train'],
                              num_workers=config['num_workers'])
    test_set = DataSet(config, 'test')
    test_loader = DataLoader(test_set,
                             collate_fn=test_set.collate,
                             batch_size=config['batch_size'],
                             shuffle=config['shuffle']['test'],
                             num_workers=config['num_workers'])
    # 训练
    train(train_loader, test_loader, config)
    # 测试
    mean_dice = test(test_loader, config, show_flag=True)
    # 保存此次实验的全部代码，并在summary中添加一行实验记录
    snapshot(config, mean_dice)


if __name__ == '__main__':
    main()
