import h5py
import numpy as np

file_y = r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\y_stim.mat"
file_sub = r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\sub_name_stim.mat"
file_chan = r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\chan.mat"

print("--- y_stim ---")
with h5py.File(file_y, 'r') as f:
    y = f['y_stim'][:]
    print(f"Shape: {y.shape}")
    for i in range(y.shape[0]):
        uniques, counts = np.unique(y[i, :], return_counts=True)
        print(f"Row {i}: {len(uniques)} unique values. Min: {y[i,:].min()}, Max: {y[i,:].max()}")
        if len(uniques) <= 60:
            print(f"  Values: {uniques}")

print("\n--- sub_name_stim ---")
with h5py.File(file_sub, 'r') as f:
    sub = f['sub_name_stim'][:]
    print(f"Shape: {sub.shape}")
    
    # sub has shape (1, 3) or (3, 1). 
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            ref = sub[i, j]
            try:
                group = f[ref]
                print(f"Group [{i},{j}] shape: {group.shape}")
                names = []
                # group is likely a dataset of references to strings
                for k in range(group.shape[0]):
                    try:
                        # check if group has 1 or 2 dims
                        if len(group.shape) == 1:
                            str_ref = group[k]
                        else:
                            str_ref = group[k, 0]
                        
                        str_obj = f[str_ref]
                        name = ''.join(chr(c[0]) for c in str_obj[:])
                        names.append(name)
                    except Exception as e:
                        print(f"Error reading string: {e}")
                print(f"  Found {len(names)} names: {names[:5]} ... {names[-5:] if len(names) > 5 else ''}")
            except Exception as e:
                print(f"Error accessing group ref: {e}")

