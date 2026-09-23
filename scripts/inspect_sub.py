import h5py

file_sub = r"c:\Users\jyoth\Downloads\ADHD_git\ADHD-EEG-Research\dataset\sub_name_stim.mat"

print("\n--- sub_name_stim ---")
with h5py.File(file_sub, 'r') as f:
    sub = f['sub_name_stim'][:]
    print(f"Shape: {sub.shape}")
    
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            ref = sub[i, j]
            try:
                group = f[ref]
                print(f"Group [{i},{j}] shape: {group.shape}")
                names = []
                
                # Iterate over the second dimension since shape is (1, N) or (N, 1)
                loop_len = group.shape[1] if group.shape[0] == 1 else group.shape[0]
                
                for k in range(loop_len):
                    try:
                        if group.shape[0] == 1:
                            str_ref = group[0, k]
                        else:
                            str_ref = group[k, 0]
                        
                        str_obj = f[str_ref]
                        name = ''.join(chr(c[0]) for c in str_obj[:])
                        names.append(name)
                    except Exception as e:
                        pass
                print(f"  Found {len(names)} names: {names[:3]} ... {names[-3:] if len(names) > 3 else ''}")
            except Exception as e:
                print(f"Error accessing group ref: {e}")
