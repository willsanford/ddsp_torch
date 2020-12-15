#
# Data
#

import torch
import os

s = os.sep


def create_data_files():
    pass


def dl_from_path(path, f_name):
    '''
    Takes a path to *.pk files and loads then into a dataloader

    Args:
      path: path to the dataloader file. Path should contain a single file
    '''
    return torch.load(os.path.join(path, f_name))
