import scipy.io as sio
import h5py
import numpy as np

def extract_strings_from_matlab_object_array(obj_array):
    result = []
    for item in obj_array.flatten():
        if isinstance(item, np.ndarray) and item.size > 0:
            if isinstance(item[0], str):
                result.append(item[0])
            elif isinstance(item.flatten()[0], str):
                result.append(item.flatten()[0])
            else:
                result.append(str(item))
        else:
            result.append(str(item))
    return result

def inspect_metadata():
    print("="*50)
    print("Inspecting chan.mat")
    mat = sio.loadmat('dataset/chan.mat', squeeze_me=True)
    chan = mat['chan']
    print(f"Type: {type(chan)}")
    print(f"Shape: {chan.shape}")
    print(f"Dtype: {chan.dtype}")
    strings = extract_strings_from_matlab_object_array(chan)
    print(f"All values: {strings}")
    print(f"Count: {len(strings)}")

    print("\n" + "="*50)
    print("Inspecting sub_name_stim.mat")
    try:
        with h5py.File('dataset/sub_name_stim.mat', 'r') as f:
            sub = f['Sub_name_stim'][:] if 'Sub_name_stim' in f else f['sub_name_stim'][:]
            print(f"Type: h5py.Dataset")
            print(f"Shape: {sub.shape}")
            print(f"Dtype: {sub.dtype}")
            # h5py object arrays contain references to strings
            strings = []
            for ref in sub.flatten():
                # dereference
                obj = f[ref]
                print(f"Ref obj shape: {obj.shape}, dtype: {obj.dtype}")
                # if it's an array of references (e.g. cell array of cell arrays)
                if obj.dtype == 'object':
                    for sub_ref in np.array(obj).flatten():
                        sub_obj = f[sub_ref]
                        try:
                            # Try to decode as UTF-16
                            s = ''.join(chr(c[0]) if obj.ndim == 2 else chr(c) for c in sub_obj[:])
                            strings.append(s)
                        except:
                            strings.append(str(sub_obj[:]))
                else:
                    try:
                        s = ''.join(chr(c[0]) if obj.ndim == 2 else chr(c) for c in obj[:])
                        strings.append(s)
                    except:
                        strings.append(str(obj[:]))
            print(f"Sample values (first 20): {strings[:20]}")
            print(f"Count: {len(strings)}")
            unique_subs = sorted(list(set(strings)))
            print(f"Unique subjects: {len(unique_subs)}")
            print(f"Unique subject list (first 10): {unique_subs[:10]}")
    except Exception as e:
        print(f"Could not load sub_name_stim.mat: {e}")

    print("\n" + "="*50)
    print("Inspecting y_stim.mat")
    with h5py.File('dataset/y_stim.mat', 'r') as f:
        y_stim = f['y_stim'][:]
        print(f"Type: h5py.Dataset")
        print(f"Shape: {y_stim.shape}")
        print(f"Dtype: {y_stim.dtype}")
        print(f"Min: {np.min(y_stim)}, Max: {np.max(y_stim)}")
        print(f"Unique values count (row 0): {len(np.unique(y_stim[0,:]))}")
        print(f"Unique values count (row 1): {len(np.unique(y_stim[1,:]))}")
        print(f"Unique values count (row 2): {len(np.unique(y_stim[2,:]))}")
        print(f"Unique values count (row 3): {len(np.unique(y_stim[3,:]))}")
        if y_stim.shape[0] >= 4:
            print("First 10 columns:")
            print(y_stim[:, :10])

if __name__ == "__main__":
    inspect_metadata()
