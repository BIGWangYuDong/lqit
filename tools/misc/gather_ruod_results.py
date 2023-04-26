import argparse
import glob
import json
import os.path as osp
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather RUOD Detection Results')
    parser.add_argument('root', help='saving path of log and checkpoint')
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
    default_ckpk_dir = [
        'faster-rcnn_r50_fpn_1x_ruod',
        'cascade-rcnn_r50_fpn_1x_ruod',
        'retinanet_r50_fpn_1x_ruod',
        'fcos_r50-caffe_fpn_gn-head_1x_ruod',
        'atss_r50_fpn_1x_ruod',
        'tood_r50_fpn_1x_ruod',
        'ssd300_120e_ruod',
    ]

    map_dict = {
        'atss_r50_fpn_1x_ruod': 'ATSS',
        'cascade-rcnn_r50_fpn_1x_ruod': 'Cascade R-CNN',
        'faster-rcnn_r50_fpn_1x_ruod': 'Faster R-CNN',
        'fcos_r50-caffe_fpn_gn-head_1x_ruod': 'FCOS',
        'retinanet_r50_fpn_1x_ruod': 'RetinaNet',
        'ssd300_120e_ruod': 'SSD',
        'tood_r50_fpn_1x_ruod': 'TOOD',
    }

    for ckpt_dir in default_ckpk_dir:
        ckpt_dir_path = osp.join(root, ckpt_dir)
        if not osp.exists(ckpt_dir_path):
            continue
        log_path = list(
            sorted(glob.glob(osp.join(root, ckpt_dir, '*', 'vis_data'))))[-1]
        log_json_path = list(glob.glob(osp.join(log_path, '*_*.json')))[-1]

        log_dict = load_json_log(log_json_path)
        max_key = max(list(log_dict.keys()))
        bbox_AP = log_dict[max_key].get('bbox_mAP')
        assert bbox_AP is not None and len(bbox_AP) == 1
        bbox_mAP = bbox_AP[0]
        name = map_dict[ckpt_dir]
        print(name, round(bbox_mAP * 100, 1))


if __name__ == '__main__':
    main()
