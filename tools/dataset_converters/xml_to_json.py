"""Convert XML (VOC) style annotations to coco format.

Note: If the image does not have annotations, it will be filtered.

Examples:
    python tools/dataset_converters/xml_to_json.py \
    ${DATASET_NAME} \
    ${PATH TO XML ANNOTATIONS} \
    ${PATH TO IMAGES} \
    ${ANNOTATIONS FILE} \
    --out-dir ${OUTPUT FILE PATH} \
    --img-suffix ${IMAGE SUFFIX}
"""
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from typing import List

import numpy as np
from mmcv import imread
from mmengine.fileio import dump, isdir, isfile, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress

from lqit.detection.datasets import get_classes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert XML (VOC) style annotations to coco format')
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument(
        'xml_path', help='The path of directory that saving xml file.')
    parser.add_argument(
        'img_path', help='The path of directory that saving images.')
    parser.add_argument('ann_file', help='Annotation file path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--img-suffix', default='jpg', help='The image suffix')
    args = parser.parse_args()
    return args


def cvt_annotations(xml_path: str, img_path: str, ann_file: str,
                    dataset_name: str, out_file: str, img_suffix: str) -> None:
    """Convert XML (VOC) style dataset to coco format.

    Args:
        xml_path (str): The path of directory that saving xml file.
        img_path (str): The path of directory that saving images.
        ann_file (str): Annotation file path.
        dataset_name (str): The dataset name.
        out_file (str): The saving file name.
        img_suffix (str): The image suffix
    """
    annotations = []
    img_names = list_from_file(ann_file)

    xml_paths = [
        osp.join(xml_path, f'{img_name}.xml') for img_name in img_names
    ]
    img_paths = [
        f'{img_path}/{img_name}.{img_suffix}' for img_name in img_names
    ]
    part_annotations = track_progress(
        parse_xml,
        list(
            zip(xml_paths, img_paths,
                [dataset_name for _ in range(len(xml_paths))])))
    annotations.extend(part_annotations)

    annotations = cvt_to_coco_json(
        annotations=annotations, dataset_name=dataset_name)
    dump(annotations, out_file)


def parse_xml(xml_path: str, img_path: str, dataset_name: str) -> dict:
    """Parse xml annotation.

    Args:
        xml_path (str): The xml file path.
        img_path (str): The image path.
        dataset_name (str): The dataset name.

    Returns:
        dict: The annotation of the current image.
    """

    dataset_class = get_classes(dataset_name)
    label_ids = {name: i for i, name in enumerate(dataset_class)}

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is not None:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
    else:
        img = imread(img_path, backend='cv2')
        h, w = img.shape[:2]
        del img

    # img_path will only keep image name
    img_path = osp.split(img_path)[-1]

    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in dataset_class:
            continue
        label = label_ids[name]
        difficult = obj.find('difficult')
        difficult = 0 if difficult is None else int(difficult.text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    if bboxes.shape[0] == 0 and bboxes_ignore.shape[0] == 0:
        print(f'\n Filter {img_path} because it does not have annotations. ')
        return None
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_to_coco_json(annotations: List[dict], dataset_name: str) -> dict:
    """Convert to coco format.

    Args:
        annotations (List[dict]): A list of annotations.
        dataset_name (str): The dataset name.

    Returns:
        dict: COCO format dictionary.
    """
    dataset_class = get_classes(dataset_name)
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        """Process annotation item."""
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(dataset_class):
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        if ann_dict is None:
            continue
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def main():
    args = parse_args()

    # get dataset name
    dataset_name = args.dataset

    # check xml file path, image file path, and ann file is exist
    xml_path = args.xml_path
    img_path = args.img_path
    ann_file = args.ann_file
    assert isdir(xml_path)
    assert isdir(img_path)
    assert isfile(ann_file) and ann_file.endswith('txt')

    # create out_dir if set out_dir, else equal to xml_path
    out_dir = args.out_dir if args.out_dir else xml_path
    mkdir_or_exist(out_dir)

    # set out_file name
    out_file = f'{osp.split(ann_file)[-1][:-4]}.json'
    out_file = osp.join(out_dir, out_file)

    print(f'processing {dataset_name}')
    print(f'Convert xml style annotation file {ann_file} to coco style')

    img_suffix = args.img_suffix

    # concert dataset
    cvt_annotations(
        xml_path=xml_path,
        img_path=img_path,
        ann_file=ann_file,
        dataset_name=dataset_name,
        out_file=out_file,
        img_suffix=img_suffix)

    print(f'Done. The file is saving at {out_file}')


if __name__ == '__main__':
    main()
