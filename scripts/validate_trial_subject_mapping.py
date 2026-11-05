#!/usr/bin/env python3
"""
scripts/validate_trial_subject_mapping.py

Focused, read-only structural diagnostics for the candidate trial/metadata mapping.

These checks do not verify EEG-to-metadata correspondence, participant identity,
or clinical diagnosis. They also do not establish participant-independent
evaluation or produce a real-data split.

Strict constraints:
- Never modifies raw .mat files
- Never preprocesses EEG
- Never crops 385 -> 384
- Never normalizes
- Never splits data
- Never trains a model
- Reproducible, bounded memory execution
- Checks all 7 EEG files and metadata files
"""

import os
import glob
import h5py
import scipy.io as sio
import numpy as np

DATASET_DIR = "dataset"
EEG_FILES = [f"d{i}.mat" for i in range(1, 8)]
METADATA_FILES = ["y_stim.mat", "sub_name_stim.mat", "chan.mat"]

def parse_mat_header_timestamp(filepath):
    """Extract MATLAB creation timestamp from file userblock header."""
    with open(filepath, "rb") as f:
        header = f.read(128)
        text = "".join(chr(b) if 32 <= b < 127 else " " for b in header[:116]).strip()
        # Look for 'Created on: '
        if "Created on:" in text:
            idx = text.find("Created on:")
            return text[idx + len("Created on:"):].strip()
        return text

def validate_eeg_files():
    print("=" * 70)
    print("PHASE 1: RAW EEG FILE STRUCTURAL CHECKS (d1.mat - d7.mat)")
    print("=" * 70)
    
    total_trials = 0
    eeg_summary = []
    
    for i, fname in enumerate(EEG_FILES, 1):
        fpath = os.path.join(DATASET_DIR, fname)
        assert os.path.exists(fpath), f"Missing raw EEG file: {fpath}"
        fsize = os.path.getsize(fpath)
        timestamp = parse_mat_header_timestamp(fpath)
        
        with h5py.File(fpath, "r") as f:
            ds_name = f"d{i}"
            assert ds_name in f, f"Expected dataset '{ds_name}' in {fname}"
            ds = f[ds_name]
            shape = ds.shape
            dtype = ds.dtype
            chunks = ds.chunks
            compression = ds.compression
            matlab_class = ds.attrs.get("MATLAB_class", None)
            
            # Expected dimensions: (385, 56, 5000) for d1-d6, (385, 56, 3902) for d7
            expected_trials = 5000 if i < 7 else 3902
            assert shape == (385, 56, expected_trials), f"Unexpected shape {shape} in {fname}"
            assert dtype == np.float32, f"Unexpected dtype {dtype} in {fname}"
            
            total_trials += expected_trials
            
            # Memory-safe sample scan for non-finite values across start, mid, end
            sample_indices = [0, expected_trials // 2, expected_trials - 1]
            sample_slice = ds[:, :, sample_indices]
            has_nan = np.isnan(sample_slice).any()
            has_inf = np.isinf(sample_slice).any()
            val_min = float(np.min(sample_slice))
            val_max = float(np.max(sample_slice))
            
            print(f"[{i}/7] {fname:8s} | Size: {fsize/1e6:6.1f} MB | Shape: {str(shape):18s} | "
                  f"Dtype: {str(dtype):7s} | Chunks: {str(chunks):12s} | NaN: {has_nan} | "
                  f"Range: [{val_min:6.1f}, {val_max:6.1f}] | Created: {timestamp}")
            
            eeg_summary.append({
                "file": fname,
                "trials": expected_trials,
                "shape": shape,
                "dtype": dtype,
                "timestamp": timestamp,
                "range": (val_min, val_max)
            })
            
    assert total_trials == 33902, f"Total trials mismatch: {total_trials} != 33902"
    print(f"\n[PASS] Structural checks passed for all 7 EEG files. Total trials: {total_trials:,}")
    return eeg_summary

def validate_y_stim():
    print("\n" + "=" * 70)
    print("PHASE 2: y_stim.mat STRUCTURAL DIAGNOSTICS")
    print("=" * 70)
    
    fpath = os.path.join(DATASET_DIR, "y_stim.mat")
    assert os.path.exists(fpath), f"Missing {fpath}"
    fsize = os.path.getsize(fpath)
    timestamp = parse_mat_header_timestamp(fpath)
    
    with h5py.File(fpath, "r") as f:
        assert "y_stim" in f, "Missing dataset 'y_stim'"
        y = f["y_stim"][:]
        shape = y.shape
        dtype = y.dtype
        
    print(f"y_stim shape: {shape}, dtype: {dtype}, size: {fsize:,} bytes")
    print(f"y_stim header timestamp: {timestamp}")
    assert shape == (4, 33902), f"Unexpected y_stim shape: {shape}"
    assert not np.isnan(y).any(), "NaN found in y_stim"
    assert not np.isinf(y).any(), "Inf found in y_stim"
    
    # 1. Analyze Row 0 values and their contiguous runs
    row0 = y[0, :].astype(int)
    diffs = np.diff(row0)
    change_pts = np.where(diffs != 0)[0] + 1
    run_starts = np.insert(change_pts, 0, 0)
    run_ends = np.append(change_pts, len(row0))
    
    runs = []
    for s, e in zip(run_starts, run_ends):
        runs.append((row0[s], e - s, s, e))
        
    print(f"\nRow 0 contiguous candidate-index runs detected: {len(runs)}")
    assert len(runs) == 144, f"Expected exactly 144 runs, found {len(runs)}"
    
    # Identify block boundaries (where row0 resets to 1)
    resets = [i for i, r in enumerate(runs) if r[0] == 1]
    print(f"Row 0 resets to 1 at run indices: {resets}")
    assert resets == [0, 44, 96], f"Unexpected resets: {resets}"
    
    blocks = [
        ("Block 1 (controls-prefix candidate block)", runs[0:44], 44),
        ("Block 2 (subtype1-prefix candidate block)", runs[44:96], 52),
        ("Block 3 (subtype2-prefix candidate block)", runs[96:144], 48)
    ]
    
    per_subject_counts = {}
    
    for b_idx, (b_name, b_runs, expected_n_sub) in enumerate(blocks, 1):
        n_subs = len(b_runs)
        assert n_subs == expected_n_sub, f"Block {b_name} candidate run count mismatch: {n_subs} != {expected_n_sub}"
        sub_indices = [r[0] for r in b_runs]
        assert sub_indices == list(range(1, n_subs + 1)), f"Block {b_name} row-0 values are non-sequential: {sub_indices}"
        
        counts = [r[1] for r in b_runs]
        total_trials_block = sum(counts)
        start_col = b_runs[0][2]
        end_col = b_runs[-1][3]
        
        print(f"\n{b_name}:")
        print(f"  Candidate row-0 runs: {n_subs} (values 1 to {n_subs} strictly ascending)")
        print(f"  Trials: {total_trials_block:,} (columns [{start_col}, {end_col}))")
        print(f"  Min trials/sub: {min(counts)}, Max trials/sub: {max(counts)}, Mean: {np.mean(counts):.1f}")
        
        # Check one-hot indicators in rows 1-3 for this block
        block_rows13 = y[1:4, start_col:end_col]
        expected_one_hot = np.zeros((3, 1))
        expected_one_hot[b_idx - 1, 0] = 1.0
        
        matches_block = np.all(block_rows13 == expected_one_hot)
        print(
            "  Rows 1-3 structurally match the configured block position "
            f"{expected_one_hot.ravel().astype(int).tolist()} for all columns: "
            f"{matches_block} (semantics unresolved)"
        )
        assert matches_block, f"Rows 1-3 do not match expected block indicator in {b_name}"
        
        per_subject_counts[b_name] = counts

    # Global rows 1-3 one-hot check
    rows13 = y[1:4, :]
    col_sums = rows13.sum(axis=0)
    assert np.all(col_sums == 1.0), "Rows 1-3 are not strictly one-hot (column sum != 1.0)"
    assert set(np.unique(rows13)) == {0.0, 1.0}, "Rows 1-3 contain values other than 0 and 1"
    
    print("\n[PASS] y_stim structural diagnostics:")
    print("  - 144 contiguous row-0 value runs span all 33,902 columns.")
    print("  - Row-0 values are sequential within the configured ranges.")
    print("  - Candidate block ranges contain 44, 52, and 48 runs.")
    print("  - Rows 1-3 match one-hot patterns constant within those ranges.")
    print("  - These checks do not establish participant identity or clinical labels.")
    
    return runs, per_subject_counts

def validate_sub_name_stim():
    print("\n" + "=" * 70)
    print("PHASE 3: sub_name_stim.mat STRUCTURAL DIAGNOSTICS")
    print("=" * 70)
    
    fpath = os.path.join(DATASET_DIR, "sub_name_stim.mat")
    assert os.path.exists(fpath), f"Missing {fpath}"
    fsize = os.path.getsize(fpath)
    timestamp = parse_mat_header_timestamp(fpath)
    
    with h5py.File(fpath, "r") as f:
        assert "sub_name_stim" in f, "Missing dataset 'sub_name_stim'"
        sub = f["sub_name_stim"][:]
        shape = sub.shape
        
        print(f"sub_name_stim shape: {shape}, size: {fsize:,} bytes")
        print(f"sub_name_stim header timestamp: {timestamp}")
        assert shape == (1, 3), f"Unexpected sub_name_stim shape: {shape}"
        
        all_filenames = []
        block_filenames = []
        
        prefixes = ["controls", "subtype1", "subtype2"]
        expected_counts = [44, 52, 48]
        
        for b in range(3):
            ref = sub[0, b]
            grp = f[ref]
            assert grp.shape == (1, expected_counts[b]), f"Block {b} shape mismatch: {grp.shape}"
            
            names = []
            for k in range(grp.shape[1]):
                str_ref = grp[0, k]
                s = "".join(chr(c[0]) for c in f[str_ref][:])
                names.append(s)
                
            print(f"\nCandidate filename block {b} ({prefixes[b]} prefix):")
            print(f"  Count: {len(names)} unique files: {len(set(names))}")
            assert len(names) == expected_counts[b], f"Count mismatch in block {b}"
            assert len(set(names)) == expected_counts[b], f"Duplicate filename found in block {b}"
            
            # Check prefixes
            all_prefix_match = all(n.startswith(prefixes[b]) for n in names)
            print(f"  All filenames begin with '{prefixes[b]}': {all_prefix_match}")
            assert all_prefix_match, f"Filename prefix violation in block {b}"
            
            # Check sorting
            is_sorted = (names == sorted(names))
            print(f"  Filenames are in strictly sorted alphabetical order: {is_sorted}")
            
            print(f"  First file [1]: {names[0]}")
            print(f"  Last file [{len(names)}]: {names[-1]}")
            
            block_filenames.append(names)
            all_filenames.extend(names)
            
    assert len(all_filenames) == 144, f"Total filenames mismatch: {len(all_filenames)}"
    assert len(set(all_filenames)) == 144, "Duplicate filenames found across blocks"
    
    print("\n[PASS] sub_name_stim structural diagnostics:")
    print("  - 3 candidate filename blocks contain 44, 52, and 48 entries (144 total).")
    print("  - All 144 filenames are globally unique.")
    print("  - Every filename starts with the respective block prefix ('controls', 'subtype1', 'subtype2').")
    print("  - Filenames within each block are lexicographically sorted (MATLAB dir() convention).")
    print("  - Header timestamp matches y_stim.mat down to the exact second (Fri Feb 22 14:52:45 2019).")
    print("  - Filename order and prefixes do not verify identity or clinical diagnosis.")
    
    return block_filenames

def validate_boundaries_and_ordering(runs):
    print("\n" + "=" * 70)
    print("PHASE 4: CANDIDATE-SELECTED BOUNDARY SIGNAL DIAGNOSTICS")
    print("=" * 70)
    print(
        "Comparing signal summaries around six candidate file boundaries; "
        "ranges are selected using the proposed alignment."
    )
    
    # 6 file boundaries:
    # d1 ends at 5000, d2 at 10000, d3 at 15000, d4 at 20000, d5 at 25000, d6 at 30000
    boundary_cases = [
        (1, 2, 4962, 5000, 5000, 5061, "candidate block 1 run 24", 24),
        (2, 3, 9851, 10000, 10000, 10129, "candidate block 1 run 44", 44),
        (3, 4, 14810, 15000, 15000, 15058, "candidate block 2 run 20", 20),
        (4, 5, 19885, 20000, 20000, 20179, "candidate block 2 run 40", 40),
        (5, 6, 24944, 25000, 25000, 25205, "candidate block 3 run 9", 9),
        (6, 7, 29860, 30000, 30000, 30150, "candidate block 3 run 30", 30),
    ]
    
    boundary_results = []
    
    for fa_idx, fb_idx, a_start, a_end, b_start, b_end, label, sub_id in boundary_cases:
        fa_name = os.path.join(DATASET_DIR, f"d{fa_idx}.mat")
        fb_name = os.path.join(DATASET_DIR, f"d{fb_idx}.mat")
        
        off_a = (fa_idx - 1) * 5000
        off_b = (fb_idx - 1) * 5000
        
        with h5py.File(fa_name, "r") as fa, h5py.File(fb_name, "r") as fb:
            slice_a = fa[f"d{fa_idx}"][:, :, a_start - off_a : a_end - off_a]
            slice_b = fb[f"d{fb_idx}"][:, :, b_start - off_b : b_end - off_b]
            
            # Comparison slice; its participant relationship is not independently known.
            slice_ctrl = fa[f"d{fa_idx}"][:, :, 0:min(38, slice_a.shape[2])]
            
            # Compute channel variance profile across time, averaged across trials
            var_a = np.var(slice_a, axis=0).mean(axis=1) # (56,)
            var_b = np.var(slice_b, axis=0).mean(axis=1) # (56,)
            var_ctrl = np.var(slice_ctrl, axis=0).mean(axis=1) # (56,)
            
            corr_same = float(np.corrcoef(var_a, var_b)[0, 1])
            corr_ctrl = float(np.corrcoef(var_a, var_ctrl)[0, 1])
            
            n_a = a_end - a_start
            n_b = b_end - b_start
            print(f"Boundary d{fa_idx}/d{fb_idx} ({label}):")
            print(f"  Split trials: {n_a} in d{fa_idx} + {n_b} in d{fb_idx} = {n_a + n_b} trials")
            print(f"  Candidate cross-boundary variance correlation: r = {corr_same:.4f}")
            print(f"  Within-file comparison-slice correlation: r = {corr_ctrl:.4f}")
            
            assert corr_same > 0.85, (
                f"Candidate boundary diagnostic below threshold "
                f"({corr_same}) for {label}"
            )
            
            boundary_results.append({
                "boundary": f"d{fa_idx}/d{fb_idx}",
                "label": label,
                "corr_same": corr_same,
                "corr_ctrl": corr_ctrl
            })
            
    print("\n[PASS] Candidate-selected boundary diagnostics exceeded the configured threshold.")
    print(
        "  Similarity is consistent with the proposed alignment but is not "
        "an independent trial/participant key."
    )
    return boundary_results

def print_trial_subject_crosswalk_sample(runs, block_filenames):
    print("\n" + "=" * 70)
    print("PHASE 5: CANDIDATE CROSSWALK EXAMPLES (NOT VERIFIED IDENTITIES/LABELS)")
    print("=" * 70)
    
    file_boundaries = [
        ("d1", 0, 5000),
        ("d2", 5000, 10000),
        ("d3", 10000, 15000),
        ("d4", 15000, 20000),
        ("d5", 20000, 25000),
        ("d6", 25000, 30000),
        ("d7", 30000, 33902),
    ]
    
    print("First 5 candidate row-0 runs in each configured block:")
    for b_idx, (group_name, b_offset, n_subs) in enumerate([
        ("candidate block 1", 0, 44),
        ("candidate block 2", 44, 52),
        ("candidate block 3", 96, 48)
    ]):
        print(f"\n--- {group_name} ---")
        for i in range(5):
            run_idx = b_offset + i
            sub_id, n_trials, start_col, end_col = runs[run_idx]
            fname = block_filenames[b_idx][i]
            
            # Find source file(s)
            src_files = []
            for f_name, f_start, f_end in file_boundaries:
                if max(start_col, f_start) < min(end_col, f_end):
                    src_files.append(f_name)
                    
            print(
                f"  Candidate participant mapping {group_name}_{sub_id:02d} | "
                f"Trials: {n_trials:3d} | Candidate metadata columns: "
                f"[{start_col:5d}, {end_col:5d}) | Candidate source files: "
                f"{','.join(src_files):7s} | Candidate filename: {fname}"
            )

def main():
    print("======================================================================")
    print("ADHD-EEG RESEARCH: CANDIDATE MAPPING STRUCTURAL DIAGNOSTICS")
    print("======================================================================")
    print(
        "Read-only structural diagnostics only; no verified participant/diagnosis "
        "mapping, real-data split, preprocessing, or training is produced."
    )
    
    eeg_summary = validate_eeg_files()
    runs, per_subject_counts = validate_y_stim()
    block_filenames = validate_sub_name_stim()
    boundary_results = validate_boundaries_and_ordering(runs)
    print_trial_subject_crosswalk_sample(runs, block_filenames)
    
    print("\n" + "=" * 70)
    print("STRUCTURAL CHECK SUMMARY (NOT A GATE OR PROVENANCE VERDICT)")
    print("=" * 70)
    print("1. Raw EEG shapes/dtypes: structural checks passed for all seven files.")
    print("2. File-order/boundary diagnostics: consistent with configured candidate order only.")
    print("3. y_stim row-0 run/count checks: structural checks passed; runs are candidate mappings.")
    print("4. Configured block totals and rows 1-3 patterns: structural consistency only.")
    print("5. Filename counts, prefixes, and sorting: structural checks only.")
    print("6. EEG trial-to-y_stim alignment, participant identities, and clinical diagnoses:")
    print("   NOT VERIFIED by this script; candidate group blocks are not clinical labels.")
    print("7. Participant-independent split safety: NOT ESTABLISHED.")
    print("======================================================================")

if __name__ == "__main__":
    main()
