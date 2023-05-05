import argparse
import os.path as osp

import mmcv
import torch
from mmdet.utils import register_all_modules as register_all_mmdet_modules
from mmengine.config import Config, DictAction
from mmengine.fileio import get, load
from mmengine.structures import InstanceData
from mmengine.utils import ProgressBar

from lqit.registry import DATASETS, VISUALIZERS
from lqit.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse detection results')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('pred_file', help='Prediction json file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Score threshold to show the prediction boxes')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--line-width',
        type=float,
        default=3.0,
        help='The linewidth of lines.')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.8,
        help='The transparency of bboxes or mask.')
    parser.add_argument(
        '--select-img',
        type=str,
        nargs='+',
        default=None,
        help='Selected image to show the result')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def load_json_results(result_path):
    det_results = load(result_path)

    outputs = []
    bbox_list = []
    score_list = []
    label_list = []
    tmp_img_id = -1
    for result in det_results:
        image_id = result['image_id']
        if image_id != tmp_img_id and tmp_img_id != -1:
            if len(bbox_list) > 0:
                bbox_tensor = torch.Tensor(bbox_list)
                score_tensor = torch.Tensor(score_list)
                label_tensor = torch.Tensor(label_list).to(torch.int)
            else:
                bbox_tensor = torch.zeros((0, 5))
                score_tensor = torch.zeros((0, ))
                label_tensor = torch.zeros((0, ), dtype=torch.int)
            pred_instances = InstanceData()
            pred_instances.bboxes = bbox_tensor
            pred_instances.scores = score_tensor
            pred_instances.labels = label_tensor
            pred_instances.set_metainfo({'image_id': tmp_img_id})
            outputs.append(pred_instances)
            bbox_list = []
            score_list = []
            label_list = []
        bbox = result['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox_list.append(bbox)
        score_list.append(result['score'])
        label_list.append(result['category_id'])
        tmp_img_id = image_id
    if len(bbox_list) > 0:
        bbox_tensor = torch.Tensor(bbox_list)
        score_tensor = torch.Tensor(score_list)
        label_tensor = torch.Tensor(label_list).to(torch.int)
    else:
        bbox_tensor = torch.zeros((0, 5))
        score_tensor = torch.zeros((0, ))
        label_tensor = torch.zeros((0, ), dtype=torch.int)
    pred_instances = InstanceData()
    pred_instances.bboxes = bbox_tensor
    pred_instances.scores = score_tensor
    pred_instances.labels = label_tensor
    pred_instances.set_metainfo({'image_id': tmp_img_id})
    outputs.append(pred_instances)

    return outputs


def load_pkl_results(pkl_path):
    results = load(pkl_path)
    outputs = []
    for result in results:
        pred_instances = InstanceData()
        assert 'pred_instances' in result
        for k, v in result['pred_instances'].items():
            pred_instances[k] = v
        outputs.append(pred_instances)
    return outputs


def main():
    args = parse_args()

    show = args.show
    output_dir = args.output_dir
    assert not (show is False and output_dir is None), \
        'Please set `--show` or `--out-dir`'

    pred_file = args.pred_file
    assert pred_file.endswith(('.pkl', '.pickle', '.json')), \
        'pred_file must be a pkl or json file'

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules, default scope is mmdet
    register_all_mmdet_modules(init_default_scope=True)
    register_all_modules(init_default_scope=False)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    if pred_file.endswith('.json'):
        det_results = load_json_results(pred_file)
    else:
        det_results = load_pkl_results(args.prediction_path)
    if len(det_results) != len(dataset):
        assert det_results[0].get('image_id') is not None
        match = True
    else:
        match = False
    progress_bar = ProgressBar(len(dataset))
    match_i = 0
    for i, item in enumerate(dataset):
        data_sample = item['data_samples']
        img_path = data_sample.img_path
        image_name = osp.basename(img_path)
        if args.select_img is not None and image_name not in args.select_img:
            progress_bar.update()
            continue

        data_sample.pred_instances = det_results[i - match_i]
        img_bytes = get(img_path)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        if match:
            result_id = det_results[i - match_i]['image_id']
            if result_id is not None:
                if result_id != data_sample.img_id:
                    match_i += 1
                    print('image index is not match')
                    continue

        if output_dir is not None:
            out_file = osp.join(output_dir, image_name)
        else:
            out_file = None
        visualizer.line_width = args.line_width
        visualizer.alpha = args.alpha
        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample=data_sample,
            draw_gt=False,
            draw_pred=True,
            show=show,
            wait_time=args.show_interval,
            pred_score_thr=args.score_thr,
            out_file=out_file)
        progress_bar.update()


if __name__ == '__main__':
    main()
