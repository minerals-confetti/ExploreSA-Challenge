import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import transform, Proj
import random

# helper func to extract a window
from rasterio.windows import Window

def convert_from_EPSG4326(xy, dataset):
    """given a list of (lat, lon) encoded in EPSG:4326, returns a list of (x, y) encoded in dataset.crs"""
    inproj, outproj = Proj('EPSG:4326'), dataset.crs

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def extract_locs(dataset_path, cachedir, csv_filepath, size=(9, 9), stride=(9, 9)):
    

class Predictor(Dataset):
    '''Creates a dataset from an input rasterio dataset and a series of valid points,
    needs columns LONGITUDE and LATITUDE
    '''

    def __init__(self, dataset_path, cachedir, csv=None, csv_filepath=None, size=(9, 9), stride=(9, 9)):
        if csv is None:

            self.locations(csv_filepath)
        else:
            self.locations = pd.read_csv(csv)

        self.lbound = None

        #checking if csv is valid

        columns = self.locations.columns
        if "LATITUDE" not in columns or "LONGITUDE" not in columns:
            raise Exception("csv invalid, LATITUDE, LONGITUDE must be in a valid csv")
        
        self.cachedir = cachedir
        self.size = size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = int(self.locations.loc[idx, "label"])
        coords = {"LATITUDE": float(self.locations.loc[idx, "LATITUDE"]), "LONGITUDE": float(self.locations.loc[idx, "LONGITUDE"])}

        dataset_dir = self.locations.loc[idx, "paths"]
        try:
            cubic = np.load("{}/{}_{}_{}.npy".format(self.cachedir, coords["LATITUDE"], coords["LONGITUDE"], dataset_dir.split("/")[-1].split(".")[0]), allow_pickle=True)
        except FileNotFoundError:
            cubic = self.extract_cubic(dataset_dir, self.conv_coords(coords, reverse=True), size=self.size)
            np.save("{}/{}_{}_{}.npy".format(self.cachedir, coords["LATITUDE"], coords["LONGITUDE"], dataset_dir.split("/")[-1].split(".")[0]), cubic)
        
        if self.lbound != None:
            cubic = self.normOP(cubic)

        cubic = torch.tensor(cubic)

        for transform in self.transform:
            cubic = transform(cubic)
        
        sample = {"label": label, "image": cubic}

        return sample
    
    def __len__(self):
        return len(self.locations)

    def extract_cubic(self, dataset_dir, coords, size=(9, 9)):
        '''takes in a dataset dir, a tuple of coord (lat, long), size of the extracted cubics, 
        and returns coords_length * size1 * size2 * n_bands'''

        with rasterio.open(dataset_dir) as dataset:
            trans_coords = convert_from_EPSG4326([coords], dataset)
            lon, lat = trans_coords[0]
            py, px = dataset.index(lon, lat)
            window = Window(px - size[0] // 2, py - size[1] // 2, size[0], size[1])
            clip = dataset.read(window=window)
            output = np.transpose(clip, (1, 2, 0))

        return np.expand_dims(output, axis=0)

    def getMinMax(self):
        max_value = self[0]["image"].numpy().max(axis=(0, 1, 2), keepdims=False)
        min_value = self[0]["image"].numpy().min(axis=(0, 1, 2), keepdims=False)

        for i in range(1, len(self)):
            temp_max = self[i]["image"].numpy().max(axis=(0, 1, 2), keepdims=False)
            temp_min = self[i]["image"].numpy().min(axis=(0, 1, 2), keepdims=False)

            max_value = np.where(temp_max > max_value, temp_max, max_value)
            min_value = np.where(temp_max < min_value, temp_min, min_value)

        self.min_value = min_value
        self.max_value = max_value

        return min_value, max_value

    def normalize(self, lbound, ubound, min_value=None, max_value=None):
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value

        self.lbound = lbound
        self.ubound = ubound

    def normOP(self, cubic):
        
        return (self.ubound - self.lbound) * (cubic - np.expand_dims(self.min_value, axis=(0, 1, 2))) / np.expand_dims((self.max_value - self.min_value), axis=(0, 1, 2)) + self.lbound

    def conv_coords(self, coords, reverse=False):
        '''converts (lat, lon) to coords used by this class'''
        if reverse:
            return coords["LATITUDE"], coords["LONGITUDE"]
        return {"LATITUDE": coords[0], "LONGITUDE": coords[1]}