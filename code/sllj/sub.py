import argparse
import re


def main(input_file, output_file, pattern, replacement):
    with open(output_file, 'w', encoding='utf-8') as file:
        with open(input_file, 'r', encoding='utf-8') as f:
            proof_set = f.read()
            write_lines = re.sub(pattern, replacement, proof_set)
            file.write(write_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input', help='input file name')
    parser.add_argument('--output', help='output file name')
    parser.add_argument('--pat', help='pattern to search for')
    parser.add_argument(
        '--rep', help='string to replace the pattern with')
    args = parser.parse_args()

    main(args.input, args.output, args.pat, args.rep)
