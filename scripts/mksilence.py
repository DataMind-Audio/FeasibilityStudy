#!/usr/bin/env python3

import argparse
import numpy as np

def db_to_gain(db):
    return 10 ** (db / 20.0)

def write_wav_header(file, sample_rate, channel_count, bits_per_sample, frame_count, is_float=False):
    format_type = 3 if is_float else 1
    bytes_per_sample = int(bits_per_sample / 8)
    block_align = int(channel_count * bytes_per_sample)
    bytes_per_sec = int(sample_rate * channel_count * bytes_per_sample)
    header_size = 44
    data_size = int(channel_count * bytes_per_sample * frame_count)
    file_size = int(data_size + header_size)

    file.write(b"RIFF")
    file.write(int(file_size - 8).to_bytes(4, 'little'))
    file.write(b"WAVE")
    file.write(b"fmt ")
    file.write((16).to_bytes(4, 'little'))
    file.write(format_type.to_bytes(2, 'little'))
    file.write(channel_count.to_bytes(2, 'little'))
    file.write(sample_rate.to_bytes(4, 'little'))
    file.write(bytes_per_sec.to_bytes(4, 'little'))
    file.write(block_align.to_bytes(2, 'little'))
    file.write(bits_per_sample.to_bytes(2, 'little'))
    file.write(b"data")
    file.write(data_size.to_bytes(4, 'little'))

    return

def write_silence_file(path, size, level):
    sample_count = int(size // 2) # 16 bit wav
    silence_range = sample_count // 3
    noise_range = sample_count - silence_range

    s16_max = 32767
    samples_left = sample_count
    with open(path, "wb") as f:
        write_wav_header(f, 44100, 1, 16, sample_count)
        while samples_left > 0:
            it = sample_count - samples_left
            chunk_size = 1024 * 1024
            if samples_left < chunk_size:
                chunk_size = samples_left

            noise = np.random.uniform(-1.0, 1.0, size=(1,chunk_size))
            if it <= silence_range:
                amp_env = 0
            else:
                start = (it - silence_range) / noise_range
                end = (it - silence_range + chunk_size) / noise_range
                amp_env = np.linspace(start, end, num=chunk_size)

            chunk = np.array(noise * amp_env * s16_max * db_to_gain(level), dtype=np.int16);
            f.write(chunk)

            samples_left -= chunk_size


def main(args):
    size = args.size * 1000 ** 3 # convert GB to bytes

    remaining = size
    file_id = 0
    while remaining > 0:
        max_wav_size = 2**32 - 1 - 44
        file_size = remaining if remaining < max_wav_size else max_wav_size
        file_name = args.output
        if file_id > 0:
            file_name = args.output.replace('.wav', '')
            file_name = f'{file_name}_{file_id}.wav'
            print(f'More than 4GiB of silence requested, creating multiple files: {file_name}')

        write_silence_file(file_name, file_size, args.level)

        remaining -= file_size
        file_id += 1

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('-s', '--size', type=float, help='size in GB of silence file', required=True)
    parser.add_argument('-o', '--output', type=str, help='output file name', required=True)
    parser.add_argument('-l', '--level', type=float, help='max level of silence in dB', default=-50)
    args = parser.parse_args()

    main(args);
