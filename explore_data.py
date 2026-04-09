import pandas as pd
import glob
import os

# Search for CSV files
base_path = r'c:\Users\carla\Desktop\EECE 798K'
csv_files = glob.glob(os.path.join(base_path, '**', '*.csv'), recursive=True)

print(f'Found {len(csv_files)} CSV files\n')

# Examine files that look like stick-slip data
sample_files = [f for f in csv_files if 'cm.csv' in f and '-checkpoint' not in f][:5]
if not sample_files:
    sample_files = csv_files[:5]

for fpath in sample_files:
    print(f'File: {os.path.basename(fpath)}')
    try:
        df = pd.read_csv(fpath, nrows=5)
        print(f'Columns: {list(df.columns)}')
        print(f'Full shape: {pd.read_csv(fpath).shape}')
        print(f'Sample data:\n{df.head(3)}')
        print('-' * 60)
    except Exception as e:
        print(f'Error: {e}\n')
