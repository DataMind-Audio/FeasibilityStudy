#!/usr/bin/env python3

import os
import sys
import shlex
import subprocess
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    assert os.path.isdir(input_dir)

    uvr_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            path = os.path.join(root, file)
            if path.lower().endswith(('.wav','.aiff', '.aif', '.mp3', '.flac')):
                uvr_files.append(path)

    subprocess.run(['python3', '-m', 'demucs', '-d', 'cuda', '-n', 'htdemucs_ft'] + uvr_files, shell=True)

    return

if __name__ == '__main__':
    main(sys.argv[1:])
