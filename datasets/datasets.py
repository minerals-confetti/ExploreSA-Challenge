import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np

class FDSSCDataset(Dataset):
    '''Creates a dataset from a given vrt dataset, and a csv of locations, and crops out a location to feed into FDSSC
    '''

    def __init__(self, data, csv, )