"""Analyze Errors based on TIDE.

Examples:
    python tools/analysis_tools/analyze_tide.py \
    ${ANNOTATIONS FILE} \
    ${RESULT JSON FILE} \
    --out ${OUT PATH}
"""
import argparse

from lqit.detection.evaluation.tide import COCO, TIDE, COCOResult


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze Errors based on TIDE')
    parser.add_argument('ann_file', help='Annotation json file path')
    parser.add_argument('pred_file', help='Prediction json file path')
    parser.add_argument(
        '--pos-thr',
        default=0.5,
        type=float,
        help='Positive threshold in TIDE')
    parser.add_argument(
        '--bkg-thr',
        default=0.1,
        type=float,
        help='Background threshold in TIDE')
    parser.add_argument(
        '--mode',
        default='bbox',
        choices=['bbox', 'mask'],
        type=str,
        help='The mode of evaluation in TIDE')
    parser.add_argument(
        '--name', default=None, type=str, help='The running name')
    parser.add_argument(
        '--out',
        default=None,
        type=str,
        help='Saving path of the TIDE result image')
    args = parser.parse_args()
    return args


def get_tide_errors(args):
    tide = TIDE(
        pos_threshold=args.pos_thr,
        background_threshold=args.bkg_thr,
        mode=args.mode)
    assert args.pred_file.endswith('json'), \
        'TIDE analyze only support json format, please set ' \
        '`CocoMetric.format_only=True` and `CocoMetric.outfile_prefix=xxx` ' \
        'to get json result first.'

    gt = COCO(path=args.ann_file, name=args.name)
    preds = COCOResult(path=args.pred_file, name=args.name)
    tide.evaluate(gt=gt, preds=preds, name=args.name)
    tide.summarize()
    all_errors = tide.all_errors
    if args.out is not None:
        tide.plot(out_dir=args.out)
        print(f'Save TIDE Image in {args.out}')
    return all_errors


def main():
    args = parse_args()
    all_errors = get_tide_errors(args)
    error_str = 'TIDE Errors: \n'
    for k, v in all_errors.items():
        error_name = f'{k}_Error'.ljust(11)
        error_str += f'{error_name}: {v:6.2f}  \n'
    print(error_str)


if __name__ == '__main__':
    main()
