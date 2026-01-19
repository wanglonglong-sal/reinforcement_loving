import shutil
import sys
from pathlib import Path

def clear_folder(dir_name):
    path = Path(dir_name)
    if path.exists():
        shutil.rmtree(path)
        print(f"[INFO]Cleared {path}")

def get_path_variables():
    entry_script = Path(sys.argv[0]).resolve()

    return entry_script.parent, entry_script.name, entry_script.stem