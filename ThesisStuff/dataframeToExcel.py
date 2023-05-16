import pandas as pd
import os, sys


# append the parent folder 'Script' for imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def make_frame2xlsx(path: str):
    frame = pd.read_pickle(path)
    frame.to_excel(path.replace('.pkl', '.xlsx'))


if __name__ == '__main__':
    path = input('Type in the path of the frame you want to make to an exel file.\n').replace('"', '')
    make_frame2xlsx(path)