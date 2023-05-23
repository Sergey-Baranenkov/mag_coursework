import os


def apply_deezer_spleeter(filename):
    os.system(f'spleeter separate -p spleeter:5stems -o audio/tmp {filename}')
