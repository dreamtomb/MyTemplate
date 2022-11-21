import os
import random
from datetime import datetime

import numpy as np
import torch
from apex import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.dataset import DataSet
from model.Network_Res2Net_GRA_NCD import Network
from record.snapshot import snapshot
from trainer.test import test
from trainer.train import train
from utils.utils import get_config, get_logger

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    # 获取所有设置
    config_path = './config/config.yaml'
    config = get_config(config_path)
    now = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
    config['now'] = now

    # 设置随机种子并指定训练显卡
    seed = config['seed']
    device = config['cuda_device']
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 设置数据集
    train_set = DataSet(config, 'train')
    train_loader = DataLoader(train_set,
                              collate_fn=train_set.collate,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle']['train'],
                              num_workers=config['num_workers'])
    test_set = DataSet(config, 'test')
    test_set.train_samples = train_set.train_samples
    test_set.test_samples = train_set.test_samples
    test_loader = DataLoader(test_set,
                             collate_fn=test_set.collate,
                             batch_size=config['batch_size'],
                             shuffle=config['shuffle']['test'],
                             num_workers=config['num_workers'])

    # 定义网络
    if config['load_model']:
        model = './checkpoints/*.pth'
        network = Network(config, channel=32)  # 使用训练好的模型继续训练
        network.load_state_dict(torch.load(model))
    else:
        network = Network(config, channel=32)  # 使用自定义初始化
    network.cuda()

    # 参数
    base, head = [], []
    for name, param in network.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print('conv1&bn1:{0}'.format(name))
        elif 'bkbone' in name:
            print('bkbone:{0}'.format(name))
            base.append(param)
        else:
            print('head:{0}'.format(name))
            head.append(param)

    # 优化器
    optimizer = torch.optim.SGD([{
        'params': base
    }, {
        'params': head
    }],
                                lr=config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay'],
                                nesterov=True)

    # 使用apex进行混合精读计算
    network, optimizer = amp.initialize(network, optimizer, opt_level='O0')

    # 创建本次实验的log、checkpoint、image_res文件夹
    log_path = '{}/{}'.format(config['log_path'], config['now'])
    image_res_path = '{}/{}'.format(config['image_res_path'], config['now'])
    checkpoints_path = '{}/{}'.format(config['checkpoints_path'],
                                      config['now'])
    record_path = '{}/{}'.format(config['record_path'], config['now'])
    os.mkdir(log_path)
    os.mkdir(image_res_path)
    os.mkdir(checkpoints_path)
    os.mkdir(record_path)

    # 使用tensorboard
    sw = SummaryWriter(log_path)

    # 使用logger进行日志记录
    log_path = '{}/log.txt'.format(log_path)
    logger = get_logger(log_path)

    # 开始迭代
    global_step = 0
    step_per_epoch = len(train_loader)
    for epoch in range(config['max_epoch']):

        # 训练
        logger.info(
            '#########################开始训练！###########################')
        global_step = train(network, train_loader, optimizer, config, sw,
                            logger, global_step, step_per_epoch, epoch)
        # 保存
        torch.save(
            network.state_dict(),
            '{}/{}/model-{}.pth'.format(config['checkpoints_path'],
                                        config['now'], epoch + 1))
        logger.info('model save to {}/{}/model-{}.pth'.format(
            config['checkpoints_path'], config['now'], epoch + 1))
        logger.info(
            '#########################训练完成！###########################')

        # 验证
        logger.info(
            '#########################开始验证！###########################')
        mean_dice = test(network, test_loader, config, None, False)
        sw.add_scalars('val_dice', {'val_dice': mean_dice},
                       global_step=global_step)
        logger.info('validation mean dice is {:.2f}'.format(mean_dice))
        logger.info(
            '#########################验证完成！###########################')
        if ((epoch + 1) % config['lr_step'] == 0):
            config['lr'] *= config['lr_schedure']
    # 测试
    logger.info('#########################开始测试！###########################')
    model_path = '{}/{}/model-{}.pth'.format(config['checkpoints_path'],
                                             config['now'],
                                             config['max_epoch'])
    mean_dice = test(network, test_loader, config, model_path, show_flag=True)
    logger.info('test mean dice is {:.2f}'.format(mean_dice))
    logger.info('#########################测试完成！###########################')
    # 保存此次实验的全部代码，并在summary中添加一行实验记录
    snapshot(config, mean_dice)


if __name__ == '__main__':
    main()
