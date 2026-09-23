import h5py
import numpy as np
import scipy.io
import os

files = [
    r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\y_stim.mat",
    r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\sub_name_stim.mat",
    r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\chan.mat"
]

def inspect_h5(file_path):
    print(f"\n{'='*50}\n--- Investigating: {os.path.basename(file_path)} (v7.3) ---\n{'='*50}")
    try:
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                val = f[key]
                print(f"\nVariable: {key}")
                print(f"Type: {type(val)}")
                if isinstance(val, h5py.Dataset):
                    print(f"Shape: {val.shape}")
                    print(f"Data type: {val.dtype}")
                    try:
                        data = val[:]
                        if data.size < 100:
                            print(f"Values: {data}")
                        if np.issubdtype(data.dtype, np.number):
                            uniques, counts = np.unique(data, return_counts=True)
                            if len(uniques) < 20:
                                print(f"Unique values and counts: {dict(zip(uniques, counts))}")
                            else:
                                print(f"Number of unique values: {len(uniques)}")
                                print(f"Min: {data.min()}, Max: {data.max()}")
                        elif val.dtype.kind == 'O':  # HDF5 object references (MATLAB cell array of strings)
                            print(f"Extracting strings from cell array...")
                            str_list = []
                            for ref in data.flatten():
                                if ref:
                                    try:
                                        obj = f[ref]
                                        str_val = ''.join(chr(c[0]) for c in obj[:])
                                        str_list.append(str_val)
                                    except Exception as e:
                                        pass
                            if str_list:
                                uniques, counts = np.unique(str_list, return_counts=True)
                                if len(uniques) < 20:
                                    print(f"Unique strings and counts: {dict(zip(uniques, counts))}")
                                else:
                                    print(f"Number of unique strings: {len(uniques)}")
                                    print(f"Sample strings: {str_list[:10]}")
                    except Exception as e:
                        print(f"Error reading data: {e}")
                elif isinstance(val, h5py.Group):
                    print(f"Group keys: {list(val.keys())}")
    except Exception as e:
        print(f"Error with h5py: {e}")

def inspect_mat(file_path):
    try:
        mat = scipy.io.loadmat(file_path, simplify_cells=True)
    except Exception as e:
        if 'v7.3' in str(e):
            inspect_h5(file_path)
            return
        else:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
            return
            
    print(f"\n{'='*50}\n--- Investigating: {os.path.basename(file_path)} ---\n{'='*50}")
    for key, val in mat.items():
        if key.startswith('__'):
            continue
        print(f"\nVariable: {key}")
        print(f"Type: {type(val)}")
        if isinstance(val, np.ndarray):
            print(f"Shape: {val.shape}")
            print(f"Data type: {val.dtype}")
            if val.size < 100:
                print(f"Values: {val}")
            if np.issubdtype(val.dtype, np.number):
                uniques, counts = np.unique(val, return_counts=True)
                if len(uniques) < 20:
                    print(f"Unique values and counts: {dict(zip(uniques, counts))}")
                else:
                    print(f"Number of unique values: {len(uniques)}")
                    print(f"Min: {val.min()}, Max: {val.max()}")
            elif val.dtype.kind in {'U', 'S', 'O'}:
                try:
                    flattened = val.flatten()
                    if len(flattened) < 1000:
                        uniques, counts = np.unique([str(x) for x in flattened], return_counts=True)
                        if len(uniques) < 20:
                            print(f"Unique values and counts: {dict(zip(uniques, counts))}")
                        else:
                            print(f"Number of unique string/obj values: {len(uniques)}")
                            print(f"Sample values: {flattened[:10]}")
                    else:
                        print(f"Too many string/obj values to count ({len(flattened)})")
                except Exception as e:
                    print(f"Error handling U/S/O type: {e}")
        elif isinstance(val, list):
            print(f"Length: {len(val)}")
            print(f"Sample values: {val[:5]}")
        elif isinstance(val, dict):
            print(f"Keys: {val.keys()}")
        else:
            print(f"Value: {val}")

for f in files:
    inspect_mat(f)
