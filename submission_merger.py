from __future__ import print_function, division
import gzip
import pandas as pd
import sys

csvs = {'/media/marat/MySSD/plankton/submissions/2015.02.04.1/2015.02.04.1.csv.gz': 1.0,
        '/media/marat/MySSD/plankton/submissions/2015.02.05.1/results.csv.gz': 2.0,
        '/media/marat/MySSD/plankton/submissions/2015.02.07.1/results.csv.gz': 1.0}

out_file_name = 'averaged'
row_names = None
new_data = None
total_w = 0
header = None


for csv, w in csvs.iteritems():
    print('Processing %s' % csv)
    with gzip.open(csv, 'rb') as f:
        table = pd.read_csv(f)
        data = table[table.columns[1:]].values
        rows_names = list(table.image.values)
        if row_names is None:
            row_names = rows_names
            new_data = data
            header = list(table.columns)
        elif row_names != rows_names:
            print('Row order doesn''t match!', file=sys.stderr)
            break
        new_data = new_data + data * w
    total_w += w

new_data = new_data / total_w

with gzip.open(out_file_name + '.csv.gz', 'wb') as f:
    f.write(','.join(header) + '\n')
    for i, fn in enumerate(row_names):
        f.write(fn + ',' + ','.join('%.8f' % prob for prob in new_data[i]) + '\n')