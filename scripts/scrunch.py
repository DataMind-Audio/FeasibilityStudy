import os
import soundfile as sf
import numpy as np 
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, help='input dataset folder', default='None')
  parser.add_argument('-p', '--percentage', type=float, help='percentage', default=50.0)
  args = parser.parse_args()

  chunk_length = 5 * 44100
  buffer = np.array([])
  percentage = round(100.0/args.percentage)
  source_folder = args.input
  out_folder = source_folder + "_scrunch"

  if not os.path.exists(out_folder):
      os.makedirs(out_folder)


  idx = 0
  for root, _, files in os.walk(source_folder):
    for file in files:
      path = os.path.join(root, file)
      chunk, sr = sf.read(path)
      buffer = np.append(buffer, chunk)
      while len(buffer) >= chunk_length:
        if idx % percentage == 0:
          sf.write(os.path.join(out_folder, str(idx) + ".wav"), buffer[:chunk_length], samplerate=44100, format="wav")
        buffer = buffer[chunk_length:]
        idx += 1
