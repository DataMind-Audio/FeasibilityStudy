# python3 -m pip install --user torch librosa soundfile absl-py

# This script takes input files and returns the processed outs with the scale and offest settings as specified

# Written by Andreas, based on Robin's batch tester


import os, time

import torch
import librosa
import soundfile

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model',
                    None,
                    help='Path to the trained RAVE model',
                    required=True)

flags.DEFINE_string('input',
                    None,
                    help='Path to directory containing test audio',
                    required=True)

flags.DEFINE_string('output',
                    None,
                    help='Output directory for the processed data',
                    required=True)

flags.DEFINE_float('scale',
                    None,
                    help='scale factors to test',
                    required=True)
flags.DEFINE_float('offset',
                    None,
                    help='offset values to test',
                    required=True)

def main(argv):
    model_path = FLAGS.model
    input_dir  = FLAGS.input
    output_dir = FLAGS.output
    scale_param = FLAGS.scale
    offset_param = FLAGS.offset

    print(f"Loading model: {model_path}")
    torch.set_grad_enabled(False) # ref ZZ
    model = torch.jit.load(model_path)

    model = model.eval()

    for file in os.listdir(input_dir):
        if not file.endswith(".wav"):
            continue

        # Load file into memory
        filepath = os.path.join(input_dir, file)
        data, sr = librosa.load(filepath, sr=44100, mono=True)
        data_len = data.shape[0]
        print(f"Loaded audio file: {file} at sample rate {sr}")

        data = torch.from_numpy(data).reshape(1, 1, -1)
        
    
        start = time.time()
        z = model.encode(data)
        z = modifyLatents(z, scale=scale_param, offset=offset_param)
        # z = torch.cat((z, z))

        y = model.decode(z).numpy()
        end = time.time()

        # Time to process model in seconds
        took = end - start
        data_duration = data_len / sr

        speed_factor = data_duration / took

        y = y[0, :, :]

        c = y.shape[0]

        # libsoundfile expects the data in this format
        y = y.transpose()

        model_name = os.path.basename(model_path)
        model_name = model_name.replace('.ts', '')

        out_name = file.replace('.wav', f"___{model_name}_scale_{scale_param}_offset_{offset_param}.wav")
        print(f"Writing processed file: {out_name} channels = {c} time = {speed_factor}x realtime")

        out_path = os.path.join(output_dir, out_name)
        soundfile.write(out_path, y, sr)

    return


def modifyLatents(z, scale=1, offset=0.0):
    return torch.add(torch.mul(z, torch.Tensor([scale])), torch.Tensor([offset]))


if __name__ == '__main__':
    app.run(main)
