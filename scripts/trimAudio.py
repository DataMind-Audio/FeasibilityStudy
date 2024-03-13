import argparse
import ffmpeg

def trim_and_convert_to_mono(input_file, output_file, end_time):
    ffmpeg.input(input_file, to=end_time).output(output_file, ac=1, format='wav').run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trim WAV file by specifying end time in seconds and convert to mono')
    parser.add_argument('--input', help='Input WAV file path', required=True)
    parser.add_argument('--output', help='Output WAV file path (trimmed and mono)', required=True)
    parser.add_argument('--end_time', type=float, help='End time for trimming in seconds', required=True)
    args = parser.parse_args()

    trim_and_convert_to_mono(args.input, args.output, args.end_time)
    print("Trimming and conversion to mono complete. Trimmed file saved as", args.output)
