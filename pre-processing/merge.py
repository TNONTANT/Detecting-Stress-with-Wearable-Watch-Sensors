import os
import pandas as pd

# Should be edit to your local environment path
COMBINED_DATA_PATH = "/Users/nontanatto/Desktop/DS/Assignment1/new_DS_ver2_includeIBI"
SAVE_PATH = "/Users/nontanatto/Desktop/DS/Assignment1/new_DS_ver2_includeIBI"

if COMBINED_DATA_PATH != SAVE_PATH:
    os.mkdir(SAVE_PATH)

print("Reading data ...")

acc, eda, hr, temp, ibi = None, None, None, None, None

signals = ['acc', 'eda', 'hr', 'temp', 'ibi']


def read_parallel(signal):
    df = pd.read_csv(os.path.join(COMBINED_DATA_PATH, f"combined_{signal}.csv"), dtype={'id': str})
    return [signal, df]

results = map(read_parallel, signals)

for i in results:
    globals()[i[0]] = i[1]

# Merge data
print('Merging Data ...')
ids = eda['id'].unique()
columns=['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'id', 'IBI', 'datetime']


def merge_parallel(id):
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)
    
    acc_id = acc[acc['id'] == id]
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)
    #bvp_id = bvp[bvp['id'] == id].drop(['id'], axis=1)
    ibi_id = ibi[ibi['id'] == id].drop(['id'], axis=1)

    # outer merge to merge every table together by datetime and it will be 
    # missing value in the row that doesn't have same datetime
    df = acc_id.merge(eda_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')
    #df = df.merge(bvp_id, on='datetime', how='outer')
    df = df.merge(ibi_id, on='datetime', how='outer')
    
    # fill missing values (NaN or None) with forward-fill and backward-fill values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df

results = map(merge_parallel, ids)

new_df = pd.concat(results, ignore_index=True)

print("Saving data ...")
new_df.to_csv(os.path.join(SAVE_PATH, "merged_data.csv"), index=False)
print('Done')