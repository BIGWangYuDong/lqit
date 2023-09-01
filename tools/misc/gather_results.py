import argparse
import glob
import json
import os
import os.path as osp
from collections import defaultdict

import pandas as pd
from rich.console import Console
from rich.table import Table


def parse_args():
    parser = argparse.ArgumentParser(description='Gather Results')
    parser.add_argument('root', help='saving root path of log and checkpoint')
    parser.add_argument(
        '--keys',
        default=[
            'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s',
            'bbox_mAP_m', 'bbox_mAP_l'
        ],
        nargs='+',
        help='keys to be gathered from log file')
    args = parser.parse_args()
    return args


def load_json_log(json_log):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dict = dict()
    with open(json_log) as log_file:
        epoch = 1
        for line in log_file:
            log = json.loads(line)
            # skip lines only contains one key
            if not len(log) > 1:
                continue
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                if '/' in k:
                    log_dict[epoch][k.split('/')[-1]].append(v)
                else:
                    log_dict[epoch][k].append(v)
            if 'epoch' in log.keys():
                epoch = log['epoch']
    return log_dict


def main():
    args = parse_args()

    root = args.root
    sub_dir = os.listdir(root)
    sub_dir.sort()

    keys = args.keys

    results_dict = defaultdict(list)

    for ckpt_dir in sub_dir:
        ckpt_dir_path = osp.join(root, ckpt_dir)
        if not osp.isdir(ckpt_dir_path):
            continue
        log_path = list(
            sorted(glob.glob(osp.join(root, ckpt_dir, '*', 'vis_data'))))[-1]
        log_json_path = list(glob.glob(osp.join(log_path, '*_*.json')))[-1]

        log_dict = load_json_log(log_json_path)
        max_key = max(list(log_dict.keys()))
        max_key_dict = log_dict.get(max_key, None)
        if max_key_dict is None:
            print(f'Warning: Cannot get results from {log_json_path}!')
            continue
        results_dict['ckpt_dir'].append(ckpt_dir)
        for key in keys:
            result = max_key_dict.get(key)
            if result is None:
                result = '-'
            else:
                if isinstance(result, list):
                    assert len(result) == 1
                    result = round(result[0] * 100, 2)
                elif isinstance(result, float):
                    result = round(result * 100, 2)
                else:
                    raise TypeError
            results_dict[key].append(str(result))

    df = pd.DataFrame(results_dict)
    saving_path = osp.join(root, 'gather_results.xlsx')
    df.to_excel(saving_path, index=False)

    print(f'Results are saved to {saving_path}')

    # print table
    title = f'Results gather from {root}'
    table = Table(title=title)
    headers = ['ckpt_dir'] + keys
    for header in headers:
        table.add_column(header)
    results_list = df.values.tolist()
    for result in results_list:
        table.add_row(*result)
    console = Console()
    console.print(table, end='')


if __name__ == '__main__':
    main()
