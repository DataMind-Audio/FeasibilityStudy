#!/usr/bin/env python3

import os, time
import argparse
import subprocess
import multiprocessing

def main(args):
    procs = []

    os.makedirs(args.output, exist_ok=True)

    log = subprocess.DEVNULL
    if args.logfile:
        log = open(args.logfile, "a")

    main_start = time.time()
    file_id = 0
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif')):
                in_path = os.path.join(root, file)
                file_no_ext = os.path.splitext(file)[0]
                out_path = os.path.join(args.output, f"{file_id:05d}_{file_no_ext}.wav")
                file_id += 1

                print(f"{in_path} -> {out_path}")

                ffmpeg_args = [
                        'ffmpeg',
                        '-i',
                        in_path,
                        '-af',
                        f'silenceremove=stop_periods=-1:stop_duration={args.duration}:stop_threshold={args.threshold}dB',
                        '-ac',
                        '1',
                        '-ar',
                        '44100',
                        '-c:a',
                        'pcm_s16le',
                        out_path,
                        '-y'
                ]

                p = subprocess.Popen(ffmpeg_args, stdout=log, stderr=log)
                procs.append(p)

                max_procs = multiprocessing.cpu_count()
                while len(procs) >= max_procs:
                    for p in procs:
                        print(f"More than {max_procs} processes running {len(procs)}, waiting for one to finish...")
                        p.wait()
                        procs.remove(p)
                        break

                for p in procs:
                    if p.poll() is not None:
                        print(f"Process {p} finished")
                        procs.remove(p)

    print(f"Waiting for {len(procs)} processes to finish...", end="")
    for p in procs:
        p.wait()

    print(" done.")

    main_end = time.time()

    print(f"Processing took {main_end - main_start:.2f} seconds")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input directory', required=True)
    parser.add_argument('-o', '--output', type=str, help="Output directory. Created if it doesn't exist", required=True)
    parser.add_argument('-d', '--duration', type=float, help='Minimum silence duration in seconds', default=3.0)
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for silence in dB', default=-60.0)
    parser.add_argument('--logfile', type=str, help='Path to log file', default=None)
    args = parser.parse_args()

    main(args)
