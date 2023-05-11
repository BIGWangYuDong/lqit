import argparse
import os

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from lqit.registry import VISUALIZERS
from lqit.utils import register_all_modules
from lqit.utils.misc import get_file_list


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--show', action='store_true', help='Show the results')
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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    register_all_modules()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.line_width = args.line_width
    visualizer.alpha = args.alpha

    # get file list
    image_list, source_type = get_file_list(args.img)

    progress_bar = ProgressBar(len(image_list))
    for image_path in image_list:
        result = inference_detector(model, image_path)

        img = mmcv.imread(image_path, channel_order='rgb')
        if source_type['is_dir']:
            filename = os.path.relpath(image_path, args.img).replace('/', '_')
        else:
            filename = os.path.basename(image_path)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # show the results
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            show=False,
            wait_time=0,
            out_file=None,
            pred_score_thr=args.score_thr)
        result_img = visualizer.get_image()
        progress_bar.update()
        if out_file:
            mmcv.imwrite(result_img[..., ::-1], out_file)

        if args.show:
            visualizer.show(result_img)

    if not args.show:
        print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
