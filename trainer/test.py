import torch

from utils.metrics import calculate_batch_dice
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
        test_num = 0
        step_per_epoch = len(loader)
        for step, (image, mask, name) in enumerate(loader):
            print('step: {}/{}'.format(step + 1, step_per_epoch))
            image = image.cuda().float()
            mask = mask.cuda().float()
            if config['model'] == 'PFSNet':
                # NOTE: 以下为PFSnet+resnet50的代码
                shape = [
                    torch.tensor(config['size']),
                    torch.tensor(config['size'])
                ]
                test_num += image.shape[0]
                pred_mask = network(image, shape=shape)  # 输出上采样到512
            else:
                # NOTE: 以下为SINet-V2+res2net50的代码
                test_num += image.shape[0]
                res5, res4, res3, pred_mask = network(image)
            pred_mask = torch.sigmoid(pred_mask)
            # 对图像进行二值化处理，大于等于0.5的置为1，其余为0
            pred_mask = binarize(pred_mask)
            temp = calculate_batch_dice(pred_mask, mask)
            if show_flag:
                show_batch_image(image, pred_mask, mask, config, name)
            dice.append(temp)
        mean_dice = torch.tensor(dice).mean()
        return mean_dice
