import datetime
import os

import torch
from apex import amp
from tensorboardX import SummaryWriter

from model.net import net
from solver.loss import structure_loss
from trainer.test import test
from utils.metrics import calculate_batch_dice
from utils.utils import binarize, get_logger


def train(train_loader, test_loader, config):
    """
    该函数直接被main函数调用，是训练函数
    """
    # 定义网络
    network = net(config, model=None)  # 使用自定义初始化
    network.train(True)
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
    network, optimizer = amp.initialize(network, optimizer, opt_level='O2')
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
    logger.info('#########################开始训练！###########################')
    global_step = 0
    step_per_epoch = len(train_loader)
    for epoch in range(config['max_epoch']):
        optimizer.param_groups[0]['lr'] = config['lr'] * 0.1
        optimizer.param_groups[1]['lr'] = config['lr']
        loss_per_ten_step = 0
        dice_per_ten_step = 0
        for step, (image, mask, name) in enumerate(train_loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            pred_mask = network(image)  # shape使用默认的None，不上采样到512
            loss = structure_loss(pred_mask, mask)
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            pred_mask = torch.sigmoid(pred_mask).detach()
            mask = mask.detach()
            pred_mask = binarize(pred_mask)  # 对图像进行二值化处理
            dice = calculate_batch_dice(pred_mask, mask)
            # 记录日志
            global_step += 1
            loss_per_ten_step += loss.item()
            dice_per_ten_step += dice
            if ((step + 1) % 100 == 0) or ((step + 1) == step_per_epoch):
                if (step + 1) % 100 == 0:
                    loss_per_ten_step = loss_per_ten_step / 100
                    dice_per_ten_step = dice_per_ten_step / 100
                else:
                    leave_steps = step_per_epoch % 100
                    loss_per_ten_step = loss_per_ten_step / leave_steps
                    dice_per_ten_step = dice_per_ten_step / leave_steps
                sw.add_scalar('lr',
                              optimizer.param_groups[1]['lr'],
                              global_step=global_step)
                sw.add_scalars('loss', {'loss': loss_per_ten_step},
                               global_step=global_step)
                sw.add_scalars('dice', {'dice': dice_per_ten_step},
                               global_step=global_step)
                logger.info(
                    '{:%Y-%m-%d_%H:%M:%S} || step:{:.0f}/{:.0f} || epoch:{:.0f}/{:.0f} || lr={:.6f} || loss={:.6f} || dice={:.6f}'
                    .format(datetime.datetime.now(), step, step_per_epoch,
                            epoch + 1, config['max_epoch'],
                            optimizer.param_groups[1]['lr'], loss_per_ten_step,
                            dice_per_ten_step))
                loss_per_ten_step = 0
                dice_per_ten_step = 0
        torch.save(
            network.state_dict(),
            '{}/{}/model-{}.pth'.format(config['checkpoints_path'],
                                        config['now'], epoch + 1))
        logger.info('model save to {}/{}/model-{}.pth'.format(
            config['checkpoints_path'], config['now'], epoch + 1))
        logger.info(
            '#########################开始验证！###########################')
        mean_dice = test(
            test_loader, config,
            '{}/{}/model-{}.pth'.format(config['checkpoints_path'],
                                        config['now'], epoch + 1), False)
        sw.add_scalars('val_dice', {'val_dice': mean_dice},
                       global_step=global_step)
        logger.info('validation mean dice is {:.2f}'.format(mean_dice))
        logger.info(
            '#########################验证完成！###########################')
        if ((epoch + 1) % config['lr_step'] == 0):
            config['lr'] *= config['lr_schedure']
    logger.info('#########################训练完成！###########################')
