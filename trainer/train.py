import datetime

import torch
# from apex import amp

from solver.loss import structure_loss
from utils.metrics import calculate_batch_dice
from utils.utils import binarize


def train(
    network,
    train_loader,
    optimizer,
    config,
    sw,
    logger,
    global_step,
    step_per_epoch,
    epoch,
):
    """
    该函数直接被main函数调用，是训练函数
    """
    network.train()
    loss_per_ten_step = 0
    dice_per_ten_step = 0
    show_step = config["show_step"]
    for step, (image, mask, name) in enumerate(train_loader):
        image, mask = image.cuda().float(), mask.cuda().float()
        if config["model"] == "PFSNet":
            # NOTE: 以下为PFSnet+resnet50的代码
            pred_mask = network(image)  # shape使用默认的None，不上采样到512
            loss = structure_loss(pred_mask, mask)
        else:
            # NOTE: 以下为SINet-V2+res2net50的代码
            preds1, preds2, preds3, pred_mask = network(image)
            loss = (
                structure_loss(preds1, mask)
                + structure_loss(preds2, mask)
                + structure_loss(preds3, mask)
                + structure_loss(pred_mask, mask)
            )
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scale_loss:
        #     scale_loss.backward()
        loss.backward()
        optimizer.step()
        pred_mask = torch.sigmoid(pred_mask).detach()
        mask = mask.detach()
        pred_mask = binarize(pred_mask)  # 对图像进行二值化处理
        dice = calculate_batch_dice(pred_mask, mask)
        # 记录日志
        global_step += 1
        loss_per_ten_step += loss.item()
        dice_per_ten_step += dice
        if ((step + 1) % show_step == 0) or ((step + 1) == step_per_epoch):
            if (step + 1) % show_step == 0:
                loss_per_ten_step = loss_per_ten_step / show_step
                dice_per_ten_step = dice_per_ten_step / show_step
            else:
                leave_steps = step_per_epoch % show_step
                loss_per_ten_step = loss_per_ten_step / leave_steps
                dice_per_ten_step = dice_per_ten_step / leave_steps
            sw.add_scalars(
                "lr",
                {
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                },
                global_step=global_step,
            )
            sw.add_scalars("loss", {"loss": loss_per_ten_step}, global_step=global_step)
            sw.add_scalars("dice", {"dice": dice_per_ten_step}, global_step=global_step)
            logger.info(
                "{:%Y-%m-%d_%H:%M:%S} || step:{:.0f}/{:.0f} || epoch:{:.0f}/{:.0f} || backbone_lr={:.7f} || head_lr={:.7f} || loss={:.6f} || dice={:.6f}".format(
                    datetime.datetime.now(),
                    step + 1,
                    step_per_epoch,
                    epoch + 1,
                    config["max_epoch"],
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    loss_per_ten_step,
                    dice_per_ten_step,
                )
            )
            loss_per_ten_step = 0
            dice_per_ten_step = 0
    return global_step
