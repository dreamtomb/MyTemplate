import torch

from utils.metrics import calculate_batch_dice, calculate_dice_per_case
from utils.utils import binarize, show_batch_image


def test(network, loader, config, model=None, show_flag=False):
    """
    该函数直接被main函数调用，是测试函数
    """
    if model is None:
        network.eval()
    else:
        network.load_state_dict(torch.load(model))
        network.eval()
    with torch.no_grad():
        dice = []
        dice_per_case = []
        test_num = 0
        step_per_epoch = len(loader)
        case = ""
        for step, (image, mask, name) in enumerate(loader):
            print(
                "\r", "step: {}/{}".format(step + 1, step_per_epoch), end="", flush=True
            )
            image = image.cuda().float()
            mask = mask.cuda().float()
            if config["model"] == "PFSNet":
                # NOTE: 以下为PFSnet+resnet50的代码
                shape = [torch.tensor(config["size"]), torch.tensor(config["size"])]
                test_num += image.shape[0]
                pred_mask = network(image, shape=shape)  # 输出上采样到512
            else:
                # NOTE: 以下为SINet-V2+res2net50的代码
                test_num += image.shape[0]
                res5, res4, res3, pred_mask = network(image)
            pred_mask = torch.sigmoid(pred_mask)
            # 对图像进行二值化处理，大于等于0.5的置为1，其余为0
            pred_mask = binarize(pred_mask)
            if step == 0:
                pred_per_case = pred_mask
                mask_per_case = mask
                case = name[0].split("/")[3]
            else:
                if case == name[0].split("/")[3]:
                    pred_per_case = torch.cat((pred_per_case, pred_mask), 0)
                    mask_per_case = torch.cat((mask_per_case, mask), 0)
                else:
                    res = calculate_dice_per_case(pred_per_case, mask_per_case)
                    dice_per_case.append(res)
                    pred_per_case = pred_mask
                    mask_per_case = mask
                    case = name[0].split("/")[3]
            temp = calculate_batch_dice(pred_mask, mask)
            if show_flag:
                show_batch_image(image, pred_mask, mask, config, name)
            dice.append(temp)
        mean_dice = torch.tensor(dice).mean()
        mean_dice_per_case = torch.tensor(dice_per_case).mean()
        print('\r')
        return mean_dice, mean_dice_per_case
