import scipy.io
import pandas as pd
import sys
import os

def convert_mat_to_csv(mat_file):
    print(f"Loading {mat_file}...")
    try:
        mat = scipy.io.loadmat(mat_file)
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return

    base_name = os.path.splitext(mat_file)[0]
    
    for key, value in mat.items():
        # Ignore MATLAB metadata keys that start with '__'
        if not key.startswith('__'):
            try:
                # Flatten or handle arrays depending on their shape
                df = pd.DataFrame(value)
                csv_filename = f"{base_name}_{key}.csv"
                df.to_csv(csv_filename, index=False)
                print(f"Successfully saved variable '{key}' to {csv_filename}")
            except Exception as e:
                print(f"Could not save variable '{key}' to CSV. It might be a complex or multi-dimensional array. Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mat_to_csv.py <path_to_mat_file.mat>")
    else:
        for file in sys.argv[1:]:
            convert_mat_to_csv(file)
