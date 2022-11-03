"""Get image metas on a specific dataset.

Examples:
    python tools/misc/get_image_metas.py \
    ${ANNOTATIONS FILE} \
    ${PATH TO IMAGES} \
    --out-dir ${OUTPUT FILE PATH} \
    --img-suffix ${IMAGE SUFFIX}
"""
import argparse
import os.path as osp
from multiprocessing import Pool

from mmcv import imread
from mmengine.fileio import dump, isdir, isfile, list_from_file
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(description='Collect image metas')
    parser.add_argument('ann_file', help='Dataset annotation file path')
    parser.add_argument('img_path', help='Dataset image file path')
    parser.add_argument(
        '--out-dir', default='data/tempImageMetas', help='output path')
    parser.add_argument('--img-suffix', default='jpg', help='The image suffix')
    parser.add_argument(
        '--nproc',
        default=4,
        type=int,
        help='Processes used for get image metas')
    args = parser.parse_args()
    return args


def get_image_metas(ann_file, img_path, out_file, img_suffix, nproc):
    image_metas = []
    img_names = list_from_file(ann_file)

    img_paths = [
        f'{img_path}/{img_name}.{img_suffix}' for img_name in img_names
    ]

    pool = Pool(nproc)

    part_image_metas = pool.starmap(get_simge_image_meta, zip(img_paths))
    image_metas.extend(part_image_metas)

    image_metas = cvt_list_to_dict(image_metas=image_metas)
    dump(image_metas, out_file)


def get_simge_image_meta(img_path):

    img = imread(img_path, backend='cv2')
    shape = img.shape

    del img
    filename = osp.join(
        osp.split(osp.split(img_path)[0])[-1],
        osp.split(img_path)[-1])
    img_meta = dict(filename=filename, ori_shape=shape)
    # img_path will only keep image name
    return img_meta


def cvt_list_to_dict(image_metas):
    image_metas_dict = {}
    for image_meta in image_metas:
        assert image_meta['filename'] not in image_metas_dict
        image_metas_dict[image_meta['filename']] = image_meta['ori_shape']
    return image_metas_dict


def main():
    args = parse_args()

    # check image file path, and ann file is exist
    img_path = args.img_path
    ann_file = args.ann_file
    assert isdir(img_path)
    assert isfile(ann_file) and ann_file.endswith('txt')

    # create out_dir if set out_dir, else equal to xml_path
    out_dir = args.out_dir
    mkdir_or_exist(out_dir)

    # set out_file name
    out_file = f'{osp.split(ann_file)[-1][:-4]}-image-metas.pkl'
    out_file = osp.join(out_dir, out_file)

    print('processing ...')
    img_suffix = args.img_suffix

    # concert dataset
    get_image_metas(
        img_path=img_path,
        ann_file=ann_file,
        out_file=out_file,
        img_suffix=img_suffix,
        nproc=args.nproc)

    print(f'Done. The file is saving at {out_file}')


if __name__ == '__main__':
    main()
