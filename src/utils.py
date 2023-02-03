"""Module with generic utility funcs."""

import os
import json


def os_list_dir_filetype(directory: str, file_type: str) -> list:
    """Return the files of file_type using os.listdir()"""
    files = [i for i in os.listdir(directory) if i.lower().endswith(file_type)]
    return files


def get_txt_file_as_str(file: str) -> str:
    """Read file and return as str."""
    with open(file, 'r', encoding='utf-8') as f:
        string = f.read()
    return string 


def write_str_to_txt_file(file_string: str, file_name: str) -> None:
    """Write file_string to file_name."""
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(file_string)


def get_json_file_as_dict(file: str) -> dict:
    """Return JSON file as dict."""
    assert file[-5:].lower() == '.json', f'{file} is not a JSON file.'
    with open(file, 'r', encoding='utf-8') as f:
        d = json.load(f)
        return d


def write_dict_to_json_file(file: str, data: dict) -> None:
    """Write data to file asn a JSON file."""
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=2)


def list_files(startpath: str) -> None:
    """Pretty print all files in a directory"""
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')