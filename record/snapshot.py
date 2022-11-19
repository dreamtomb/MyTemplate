import os
import shutil

import pandas as pd


def find_all_files_with_specified_suffix(target_dir='./',
                                         target_suffix=[
                                             '.py', '.yaml', '.json'
                                         ]):
    """
    该函数找到当前路径下所有文件后缀名满足要求的文件的路径

    Args:
        target_dir (str, optional): 查找后缀名的目标路径. Defaults to './'.
        target_suffix (list, optional): 后缀名列表. Defaults to [ '.py', '.yaml', '.json' ].

    Returns:
        find_res (list): 查找到的拥有目标后缀名的所有文件的路径 
    """
    find_res = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name in target_suffix:
                find_res.append(os.path.join(root_path, file))
    return find_res


def snapshot(config, mean_dice):
    """
    保存此次实验的全部代码，并在summary中添加一行实验记录

    Args:
        config (dict): 配置信息
    """
    # 定义路径
    record_path = '{}/{}'.format(config['record_path'], config['now'])
    summary_path = config['summary_path']
    # 将所有代码和配置文件保存到record
    all_files = find_all_files_with_specified_suffix('./',
                                                     ['.py', '.yaml', '.json'])
    for item in all_files:
        shutil.copy(item, record_path + '/' + item.split('/')[-1])
    # 在summary表格中添加一行实验记录
    summary_data = pd.read_excel(summary_path)
    new_data = {
        summary_data.columns[0]: [config['now']],
        summary_data.columns[1]: ['归一化\r随机翻转'],
        summary_data.columns[2]: [' '],
        summary_data.columns[3]: [
            '{}'.format(config).replace('\'', '').replace('{', '').replace(
                '}', '').replace(',', '\r')
        ],
        summary_data.columns[4]: ['{}'.format(mean_dice)],
        summary_data.columns[5]:
        ['{}/{}'.format(config['checkpoints_path'], config['now'])],
        summary_data.columns[6]:
        ['{}/{}'.format(config['record_path'], config['now'])],
        summary_data.columns[7]:
        ['{}/{}'.format(config['log_path'], config['now'])],
        summary_data.columns[8]:
        ['{}/{}/log.txt'.format(config['log_path'], config['now'])],
    }
    new_data = pd.DataFrame(new_data)
    summary_data = pd.concat([summary_data, new_data], axis=0)
    summary_data.to_excel(summary_path, index=False)
    print('snapshot successful!')
