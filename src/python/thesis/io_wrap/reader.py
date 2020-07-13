import csv
import os


def read_text_file(path):
    with open(path, 'r') as file:
        return list(file.readlines())


def read_csv_file_by_path(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)


def read_csv_file(name, dest):
    with open(dest + '/' + name, 'r') as file:
        reader = csv.reader(file)
        return list(reader)


def read_files_in_directory(dest, title=None):
    files = []
    for entry in os.scandir(dest):
        file = read_csv_file(name=entry.name, dest=dest)
        if title is None or title.compare(file[0]):
            files.append(file)

    return files
