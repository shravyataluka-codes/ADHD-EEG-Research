import sys
import scipy.io

file_path = r'c:\projects\ADHD-EEG-Research\dataset\d1.mat'

try:
    # Try scipy.io first (for older MATLAB versions up to v7.2)
    print("Trying scipy.io.loadmat...")
    mat = scipy.io.whosmat(file_path)
    print("Format: MATLAB pre-v7.3 (scipy.io)")
    for var_info in mat:
        name, shape, dtype = var_info
        print(f"Variable: {name}, Shape: {shape}, Dtype: {dtype}")
        
        # Read a small sample
        var_data = scipy.io.loadmat(file_path, variable_names=[name])[name]
        print(f"Sample values (first few): {var_data.flatten()[:5]}")
        print("-" * 40)
except NotImplementedError:
    print("NotImplementedError: Likely MATLAB v7.3+ format, trying h5py...")
    import h5py
    try:
        with h5py.File(file_path, 'r') as f:
            print("Format: MATLAB v7.3 (HDF5)")
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}, Compression: {obj.compression}, Chunks: {obj.chunks}")
                    if obj.size > 0:
                        try:
                            # if 3D, read some slice
                            if len(obj.shape) == 3:
                                print(f"Sample values (0, 0, 0:5): {obj[0, 0, :5]}")
                            elif len(obj.shape) == 2:
                                print(f"Sample values (0, 0:5): {obj[0, :5]}")
                            elif len(obj.shape) == 1:
                                print(f"Sample values (0:5): {obj[:5]}")
                        except Exception as e:
                            print(f"Could not read sample: {e}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            f.visititems(print_attrs)
            print("-" * 40)
    except Exception as e:
        print(f"Failed to read with h5py: {e}")
except Exception as e:
    print(f"Failed to read file: {e}")
