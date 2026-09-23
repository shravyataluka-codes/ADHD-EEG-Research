import h5py
import numpy as np

file_y = r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\y_stim.mat"

with h5py.File(file_y, 'r') as f:
    y = f['y_stim'][:]
    
    # Check row 0 distribution
    print("Row 0 unique values and their frequencies:")
    uniques, counts = np.unique(y[0, :], return_counts=True)
    print(dict(zip(uniques, counts)))
    
    # Check rows 1-3 combinations
    print("\nCombinations of Rows 1, 2, 3:")
    combinations = y[1:4, :]
    # transpose to shape (33902, 3)
    combinations = combinations.T
    unique_rows, counts = np.unique(combinations, axis=0, return_counts=True)
    for r, c in zip(unique_rows, counts):
        print(f"Combination {r}: count {c}")
        
    # Check relation between Row 0 (subject index?) and the combinations
    print("\nDo specific Row 0 values strictly correspond to specific Row 1-3 combinations?")
    for r, c in zip(unique_rows, counts):
        # find where this combination occurs
        mask = np.all(combinations == r, axis=1)
        sub_ids = np.unique(y[0, mask])
        print(f"Combination {r} covers {len(sub_ids)} unique Row 0 values. Max Row 0 value here: {sub_ids.max() if len(sub_ids)>0 else 'N/A'}")
