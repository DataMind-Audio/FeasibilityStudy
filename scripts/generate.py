import torch
import librosa as li
import soundfile as sf
from absl import app, flags
import os, time

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                          None,
                          help='Path to an audio file',
                          required=True)
flags.DEFINE_string('output_path',
                    None,
                    help='Output directory for the dataset',
                    required=True)
flags.DEFINE_string('model',
                    None,
                    help='Path to the trained RAVE model',
                    required=True)

def main(argv):
    print('loading model')
    torch.set_grad_enabled(False)
    model = torch.jit.load(FLAGS.model).eval()
    print('loading audio')
    print(FLAGS.input_path)
    x, sr = li.load(FLAGS.input_path)
    print(f'encoding {sr}Hz audio through model')
    x = torch.from_numpy(x).reshape(1,1,-1)
    z = model.encode(x)
    z[:, 0] += torch.linspace(-2,2,z.shape[-1])
    print('decoding latents')
    y = model.decode(z).numpy().reshape(-1)
    print('saving wav')
    fname = f'{FLAGS.input_path.split("/")[-1]}X{FLAGS.model.split("/")[-1].split(".")[0]}{time.strftime("%Y-%m-%d-%H.%M")}.wav'
    sf.write(os.path.join(FLAGS.output_path, fname), y, sr)

if __name__ == '__main__':
    app.run(main)