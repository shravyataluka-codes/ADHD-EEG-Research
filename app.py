"""
================================================================================
ADHD EEG RESEARCH PROTOTYPE — ROOT LAUNCHER
================================================================================
Allows running the prototype directly from repository root:
    streamlit run app.py
or
    streamlit run prototype/app.py
================================================================================
"""

import sys
import runpy
from pathlib import Path

# Add repository root to python search path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

prototype_script = repo_root / "prototype" / "app.py"

if __name__ == "__main__":
    runpy.run_path(str(prototype_script), run_name="__main__")
