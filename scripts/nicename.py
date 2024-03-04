#!/usr/bin/env python3

import os
import re
import argparse

def nicename(path):
    nicepath = re.sub(r'[^A-Za-z0-9_./-]', '_', path)
    nicepath = re.sub(r'__*', '_', nicepath)
    nicepath = re.sub(r'^_', '', nicepath)
    return nicepath

def main(args):
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    for root, dirs, files in os.walk(input_dir):
        out_root = root.replace(input_dir, output_dir, 1)
        out_root = nicename(out_root)

        os.makedirs(out_root, exist_ok=True)

        for file in files:
            in_path = os.path.join(root, file)
            nice_file = nicename(file)

            dot_count = nice_file.count('.')
            if dot_count > 1:
                nice_file = nice_file.replace('.', '', dot_count - 1)

            out_path = os.path.join(out_root, nice_file)

            file_id = 0
            while os.path.exists(out_path):
                splitext = os.path.splitext(out_path)
                basename = splitext[0]
                ext = splitext[1]
                out_path = f'{basename}_{file_id}.{ext}'
                file_id += 1

            os.link(in_path, out_path)

            print(f'{in_path} -> {out_path}')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input directory', required=True)
    parser.add_argument('-o', '--output', type=str, help='ouput directory', required=True)
    args = parser.parse_args()

    main(args)
