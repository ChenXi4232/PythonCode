import csv
import argparse


def csv_to_markdown(csv_file, txt_file, add_index):
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        table = ''
        headers = next(csv_reader)

        if add_index:
            headers.insert(0, 'Index')

        table += '| ' + ' | '.join(headers) + ' |\n'
        table += '| ' + \
            ' | '.join(['-' for _ in range(len(headers))]) + ' |\n'

        for idx, row in enumerate(csv_reader):
            if add_index:
                row.insert(0, str(idx+1))
            table += '| ' + ' | '.join(row) + ' |\n'

    with open(txt_file, 'w', encoding='utf-8') as output:
        output.write(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert CSV to Markdown table')
    parser.add_argument('--input', type=str,
                        help='Input CSV file path', required=True)
    parser.add_argument('--output', type=str,
                        help='Output TXT file path', required=True)
    parser.add_argument('--add_index', action='store_true',
                        help='Add index column')

    args = parser.parse_args()

    csv_to_markdown(args.input, args.output, args.add_index)
