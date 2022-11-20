import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    """
    计算预测和ground truth的IoU+BCE的loss之和作为总loss(指标要用二值,loss要用概率)

    Args:
        pred (torch.tensor[bs,1,H,W]): 模型的预测结果,没有进行sigmoid以及二值化
        mask (torch.tensor[bs,1,H,W]): 数据集中的GT

    Returns:
        torch.tensor: IoU_loss+BCE_loss,是单独的数
    """
    wbce_mean = BCE_loss(pred, mask)
    iou_mean = IoU_loss(pred, mask)
    return wbce_mean + iou_mean


def dice_loss(pred, mask):
    """
    计算DICE loss

    Args:
        pred (torch.tensor[bs,1,H,W]): 模型的预测结果,没有进行sigmoid以及二值化
        mask (torch.tensor[bs,1,H,W]): 数据集中的GT

    Returns:
        torch.tensor: 一个batch的DICE的平均值,是单独的数
    """
    smooth = 1e-5
    # 计算DICE系数
    inter = (pred * mask).sum(axis=(2, 3))
    union = (pred + mask).sum(axis=(2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return dice.mean()


def IoU_loss(pred, mask):
    """
    计算IoU loss

    Args:
        pred (torch.tensor[bs,1,H,W]): 模型的预测结果,没有进行sigmoid以及二值化
        mask (torch.tensor[bs,1,H,W]): 数据集中的GT

    Returns:
        torch.tensor: 一个batch的IoU的平均值,是单独的数
    """
    pred_sigmoid = torch.sigmoid(pred)
    inter = (pred_sigmoid * mask).sum(dim=(2, 3))
    union = (pred_sigmoid + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def BCE_loss(pred, mask):
    """
    计算BCE loss

    Args:
        pred (torch.tensor[bs,1,H,W]): 模型的预测结果,没有进行sigmoid以及二值化
        mask (torch.tensor[bs,1,H,W]): 数据集中的GT

    Returns:
        torch.tensor: 一个batch的BCE的平均值,是单独的数
    """
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    return wbce.mean()
