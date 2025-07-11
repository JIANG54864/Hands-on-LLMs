from pathlib import Path
import os
from typing import Any

base_dir=Path("../../workspace")
def read_file(filename: str)->str:
    """
    Reads a file and returns a list of lines
    """
    print("Reading file: ", filename)
    try:
        with open(base_dir/filename, 'r') as f:
            lines: str = f.read()
        return lines
    except Exception as e:
        return f"Error reading file: {e}"

def list_files()->list[str]:
    """
    Lists all files in the directory
    """
    print("Listing files in directory")
    file_list: list[Any]= []
    for itrm in base_dir.rglob("*"):
        if itrm.is_file():
            file_list.append(str(itrm.relative_to(base_dir)))
    return file_list

def rename_file(old_name: str, new_name: str)->str:
    """
    Renames a file
    """
    print("Renaming file: ", old_name, " to ", new_name)
    try:
        new_path = base_dir / new_name
        if not str(new_path).startswith(str(base_dir)):
            return "Invalid new file name"
        os.makedirs(new_path.parent, exist_ok=True)
        os.rename(base_dir/old_name, new_path)
        return "File renamed successfully"
    except Exception as e:
        return f"Error renaming file: {e}"