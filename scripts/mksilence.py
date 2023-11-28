#!/usr/bin/env python3

import argparse
import numpy as np
import soundfile as sf

def db_to_gain(db):
    return 10 ** (db / 20.0)

def main(args):
    size = args.size * 1000 ** 3 # convert GB to bytes
    sample_count = int(size // 2) # 16 bit wav
    silence_range = sample_count // 3
    noise_range = sample_count - silence_range

    noise = np.random.uniform(-1.0, 1.0, noise_range)
    amp_env = np.linspace(0, db_to_gain(args.level), num=noise_range)

    noise *= amp_env

    out = np.zeros((1, silence_range))
    out = np.append(out, noise)

    sf.write(args.output, out, 44100, subtype='PCM_16')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('-s', '--size', type=float, help='size in GB of silence file', required=True)
    parser.add_argument('-o', '--output', type=str, help='output file path', required=True)
    parser.add_argument('-l', '--level', type=float, help='max level of silence in dB', default=-60)
    args = parser.parse_args()
    main(args);


