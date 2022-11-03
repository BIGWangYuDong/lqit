"""Write file into txt file.
If use `split`, the script will base on `proportion` to split the total file
to train and val set, and save `train.txt` and `val.txt` file.
If do not use `split`, the script will write all files into txt file.
Here is an example to run this script.
Example:
    python tools/misc/get_uwdata_meta.py \
    ${PATH TO ANNOTATIONS} \
    ${FILE SUFFIX} \
    --out-dir ${OUTPUT FILE PATH} \
    --name ${FILENAME} \
    --split     # split to train and val
"""

import argparse
import os

import numpy as np
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split Underwater detection data to train and val(test) '
        'data, and save to txt file')
    parser.add_argument('path', help='the path of xml style annotation')
    parser.add_argument('suffix', help='the suffix will be counted')
    parser.add_argument(
        '--name',
        default='train.txt',
        type=str,
        help='the name of saving file, will use when split is `False`')
    parser.add_argument(
        '-o', '--out-dir', default='data/txt_files', help='output path')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument(
        '--split',
        default=False,
        action='store_true',
        help='Whether split data to train and validation')
    parser.add_argument(
        '-p',
        '--proportion',
        default=1 / 5,
        type=float,
        help='Proportion of the validation to the total size, '
        'will use when split is True')

    args = parser.parse_args()
    return args


def filter_suffix(filenames, suffix):
    """filter files that not end with specific suffix."""
    valid_indexes = []
    for i, filename in enumerate(filenames):
        if filename.endswith(suffix):
            valid_indexes.append(i)
    return valid_indexes


def write_txt_file(indexes, file_path, suffix):
    with open(file_path, 'w') as f:
        for index in indexes:
            # ignore suffix
            name = index.replace(f'.{suffix}', '')
            f.write(name)
            f.write('\n')
        f.close()


def main():
    args = parse_args()
    print(f'{"-" * 5} Start Processing {"-" * 5}')
    if args.seed is None:
        seed = np.random.randint(2**31)
    else:
        seed = args.seed
    print(f'Set random seed to {seed}')
    np.random.seed(seed)

    out_dir = args.out_dir
    print(f'Save path is {out_dir}')
    mkdir_or_exist(out_dir)

    assert os.path.exists(args.path), f'{args.path} does not exists'
    filenames = os.listdir(args.path)
    filenames.sort()  # sort the list

    # filter the files that are not endswith the setting suffix
    valid_indexes = filter_suffix(filenames, args.suffix)
    filenames = [filenames[i] for i in valid_indexes]

    total_size = len(filenames)
    assert total_size > 0, 'Total size must larger than 0'

    shuffle_index = np.random.permutation(total_size)
    filenames = [filenames[i] for i in shuffle_index]
    if args.split:
        assert 0 < args.proportion < 1, \
            'Proportion of the validation to the total need ' \
            f'between 0 and 1, but get {args.proportion}'
        proportion = args.proportion
        val_size = round(proportion * total_size)
        val_filename = filenames[:val_size]

        train_size = int(total_size - val_size)
        train_filename = filenames[val_size:]
        print('-' * 10)
        print(f'Collect {total_size} files, proportion is {proportion}, '
              f'split {train_size} for training and {val_size} for validation')

        # write file
        train_file = os.path.join(out_dir, 'train.txt')
        write_txt_file(train_filename, train_file, args.suffix)
        print(f'Train txt file save to {train_file}')

        val_path = os.path.join(out_dir, 'val.txt')
        write_txt_file(val_filename, val_path, args.suffix)
        print(f'Validation txt file save to {val_path}')
        print(f'{"-" * 5} Done {"-" * 5}')
    else:
        name = args.name
        if not name.endswith('txt'):
            name = f'{name}.txt'
        file_path = os.path.join(out_dir, name)
        write_txt_file(filenames, file_path, args.suffix)
        print(f'Txt file save to {file_path}')
        print(f'{"-"*5} Done {"-"*5}')


if __name__ == '__main__':
    main()
