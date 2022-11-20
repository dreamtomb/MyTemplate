def calculate_batch_dice(pred_mask, gt_mask):
    """
    计算pred_mask和gt_mask的DICE系数(指标要用二值,loss要用概率)

    Args:
        pred_mask ( torch.Tensor[bs, 1, H, W] ): 应当为network的输出结果进行sigmoid和二值化处理之后的二值图。
        gt_mask ( torch.Tensor[bs, 1, H, W] ): 应当为二值图像,只有0和1。

    Returns:
        dice_mean : 这一个batch的数据的平均DICE系数
    """
    smooth = 1e-5
    # 计算DICE系数
    intersection = (pred_mask * gt_mask).sum(axis=(2, 3))
    union = (pred_mask + gt_mask).sum(axis=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_mean = dice.mean()
    return dice_mean
