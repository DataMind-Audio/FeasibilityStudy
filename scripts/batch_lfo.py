import os
import math
import time
import argparse

import torch
import librosa
import soundfile as sf
import numpy as np

class Oscillator:
    def __init__(self, sample_rate, freq):
        self.phase = 0
        self.delta = freq / sample_rate

    def process_block(self, frames):
        samples = [self.process() for i in range(frames)]
        return samples;

    def process(self):
        sample = math.sin(2 * math.pi * self.phase)

        self.phase += self.delta

        if self.phase > 1.0:
            self.phase -= 1.0;

        return sample

def main(args):

    model_path = args.model
    input_dir = args.input
    output_dir = args.output

    model_name = os.path.basename(model_path)

    # NOTE(robin): We don't need gradients for back prop. Disabling
    # them here should improve perfomance slightly
    torch.set_grad_enabled(False)

    model = torch.jit.load(model_path)

    model = model.eval()

    for file in os.listdir(input_dir):
        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        out_path = out_path.replace('.wav', f'__{model_name}.wav')

        x, sr = librosa.load(filepath, sr=44100, mono=True)
        n = len(x)

        print(f'File {file} has {n} samples')

        scale_lfo = Oscillator(sr, args.scale_freq)
        offset_lfo = Oscillator(sr, args.offset_freq)

        f = sf.SoundFile(out_path, 'w', sr, 1)
        scale_lfo_f = sf.SoundFile(os.path.join(output_dir, "scale_lfo.wav"), 'w', sr, 1)
        offset_lfo_f = sf.SoundFile(os.path.join(output_dir, "offset_lfo.wav"), 'w', sr, 1)

        start = time.time()

        remaining = n
        has_nan = False
        while remaining > 0:
            it = n - remaining
            block_size = 2048
            chunk_size = block_size if remaining >= block_size else remaining

            chunk = x[it : it + chunk_size];

            if chunk_size != block_size:
                padding = np.zeros(block_size - chunk_size, dtype=np.float32)
                chunk = np.concatenate((chunk, padding))

            assert(len(chunk) == block_size)

            torch_chunk = torch.from_numpy(chunk).reshape(1, 1, -1)

            z = model.encode(torch_chunk)

            scale = np.array(scale_lfo.process_block(block_size))
            offset = np.array(offset_lfo.process_block(block_size))

            scale *= args.scale_amplitude
            offset = (offset + 1.0) / 2.0 * args.offset_amplitude

            z += offset[0]
            z *= 1.0 + scale[0]

            # NOTE(robin): shape is (1, 1, block_size)
            y = model.decode(z).numpy()

            # NOTE(robin): shape is (block_size,)
            y = y[0, 0, :]

            nans_this_block = True in [math.isnan(i) for i in y]

            if nans_this_block:
                has_nan = True

            remaining -= block_size

            f.write(y)
            offset_lfo_f.write(offset)
            scale_lfo_f.write(scale)

        end = time.time()

        elapsed = end - start

        print(f'Processed at {(n / sr) / elapsed:.2f}x realtime')

        f.close()
        offset_lfo_f.close()
        scale_lfo_f.close()

        if has_nan:
            print(f'Warning: model {model_name} outputs NaNs')


    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--scale_amplitude', type=float, default=2.0)
    parser.add_argument('--scale_freq', type=float, default=0.1)
    parser.add_argument('--offset_amplitude', type=float, default=1.0)
    parser.add_argument('--offset_freq', type=float, default=0.05)
    args = parser.parse_args();

    main(args)
