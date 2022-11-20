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


def calculate_dice_per_case(pred_mask, gt_mask):
    """
    计算一个病人所有切片的pred_mask的堆叠和所有切片的gt_mask的堆叠的DICE系数(指标要用二值)

    Args:
        pred_mask ( torch.Tensor[slice_num, 1, H, W] ): 应当为network的输出结果进行sigmoid和二值化处理之后的二值图。其中slice_num是切片数。
        gt_mask ( torch.Tensor[slice_num, 1, H, W] ): 应当为二值图像,只有0和1。其中slice_num是切片数。

    Returns:
        dice_per_case : 这一个病人的所有切片的整体的dice。
    """
    smooth = 1e-5
    # 计算DICE_per_case
    intersection = (pred_mask * gt_mask).sum()
    union = (pred_mask + gt_mask).sum()
    dice_per_case = (2.0 * intersection + smooth) / (union + smooth)
    return dice_per_case
