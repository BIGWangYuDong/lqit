# Modified from https://github.com/open-mmlab/mmyolo
import argparse
import os
import urllib
from typing import Sequence

import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar, scandir

from lqit.registry import VISUALIZERS
from lqit.utils import register_all_modules

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize feature map')
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--target-layers',
        default=['backbone'],
        nargs='+',
        type=str,
        help='The target layers to get feature map, if not set, the tool will '
        'specify the backbone')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--show', action='store_true', help='Show the featmap results')
    parser.add_argument(
        '--channel-reduction',
        default='select_max',
        help='Reduce multiple channels to a single channel')
    parser.add_argument(
        '--topk',
        type=int,
        default=4,
        help='Select topk channel to show by the sum of each channel')
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        default=[2, 2],
        help='The arrangement of featmap when channel_reduction is '
        'not None and topk > 0')
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


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, img_path):
        self.activations = []
        results = inference_detector(self.model, img_path)
        return results, self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()


def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255
                         ] * (
                             image_row * image_column - img_count)
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    register_all_modules()
    channel_reduction = args.channel_reduction
    if channel_reduction == 'None':
        channel_reduction = None
    assert len(args.arrangement) == 2

    model = init_detector(args.config, args.checkpoint, device=args.device)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    if args.preview_model:
        print(model)
        print('\n This flag is only show model, if you want to continue, '
              'please remove `--preview-model` to get the feature map.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.{target_layer}'))
        except Exception as e:
            print(model)
            raise RuntimeError('layer does not exist', e)

    activations_wrapper = ActivationsWrapper(model, target_layers)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    image_list, source_type = get_file_list(args.img)

    progress_bar = ProgressBar(len(image_list))
    for image_path in image_list:
        result, featmaps = activations_wrapper(image_path)
        if not isinstance(featmaps, Sequence):
            featmaps = [featmaps]

        flatten_featmaps = []
        for featmap in featmaps:
            if isinstance(featmap, Sequence):
                flatten_featmaps.extend(featmap)
            else:
                flatten_featmaps.append(featmap)

        img = mmcv.imread(image_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')

        if source_type['is_dir']:
            filename = os.path.relpath(image_path, args.img).replace('/', '_')
        else:
            filename = os.path.basename(image_path)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        # show the results
        shown_imgs = []
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=None,
            pred_score_thr=args.score_thr)
        drawn_img = visualizer.get_image()

        for featmap in flatten_featmaps:
            shown_img = visualizer.draw_featmap(
                featmap[0],
                drawn_img,
                channel_reduction=channel_reduction,
                topk=args.topk,
                arrangement=args.arrangement)
            shown_imgs.append(shown_img)

        shown_imgs = auto_arrange_images(shown_imgs)

        progress_bar.update()
        if out_file:
            mmcv.imwrite(shown_imgs[..., ::-1], out_file)

        if args.show:
            visualizer.show(shown_imgs)

    if not args.show:
        print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


# Please refer to the usage tutorial:
# https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
if __name__ == '__main__':
    main()
