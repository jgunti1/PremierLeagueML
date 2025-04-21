import pandas as pd
import os
from glob import glob

folder_path = '/Users/jordangunti/Documents/PremierLeagueML/CSVs'  # Update if needed
csv_files = glob(os.path.join(folder_path, '*.csv'))

dataframes = []

count = 0

for f in csv_files:
    try:
        df = pd.read_csv(f, encoding='utf-8',on_bad_lines='skip')  # First try UTF-8
        count += 1
        print(f"Read {f} with utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(f, encoding='ISO-8859-1',on_bad_lines='skip')  # Fallback to latin1
            print(f"Read {f} with ISO-8859-1")
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

    if not df.empty:
        dataframes.append(df)
    else:
        print(f"Skipped empty file: {f}")

if dataframes:
    all_data = pd.concat(dataframes, ignore_index=True)
    print("Combined shape:", all_data.shape)
else:
    print("No usable CSVs found.")
print(count)
