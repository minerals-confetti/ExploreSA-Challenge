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

def extract_cubic(dataset, coords, size=(9, 9)):
    '''takes in a opened rasterio dataset, a list of tuples of coords [(lat, long),], size of the extracted cubics, 
    and returns coords_length * size1 * size2 * n_bands'''
    bands = dataset.count
    output = np.zeros((len(coords), *size, bands))

    trans_coords = convert_from_EPSG4326(coords, dataset)
    for i, (lon, lat) in enumerate(trans_coords):
        py, px = dataset.index(lon, lat)
        window = Window(px - size[0] // 2, py - size[1] // 2, size[0], size[1])
        clip = dataset.read(window=window)
        try:
            output[i] = np.transpose(clip, (1, 2, 0))
        except ValueError:
            output[i] = np.full((*size, bands), np.nan)

    return output

class LocationChecker():
    '''creates a LocationChecker from a directory of datasets, 
    and checks if a location (EPSG4326) {"LATITUDE": num, "LONGITUDE": num} is valid
    if valid, returns the dataset location, and if from multiple datasets, returns a random one
    if invaid, returns None
    '''

    def __init__(self, datadir, rigorous=True, size=(9, 9)):
        self.bounds_dict = {}
        for dataset in self.walkdir(datadir):
            with rasterio.open(dataset) as ds:
                llx, lly, urx, ury = ds.bounds
                self.bounds_dict[dataset] = self.convert_to_EPSG4326([(llx, lly), (urx, ury)], ds)
        self.rigorous = rigorous
        self.size = size

    def check(self, coords):
        '''coords: (EPSG4326) {"LATITUDE": num, "LONGITUDE": num}'''
        valid_list = []
        for dataset, bounds in self.bounds_dict.items():
            if bounds[0][0] < coords["LATITUDE"] < bounds[1][0]:
                if bounds[0][1] < coords["LONGITUDE"] < bounds[1][1]:
                    valid_list.append(dataset)

        if self.rigorous and valid_list:
            valider_list = []
            for dataset in valid_list:
                with rasterio.open(dataset) as ds:
                    sample = extract_cubic(ds, [(coords["LATITUDE"], coords["LONGITUDE"])], size=self.size)
                    if not np.isnan(np.sum(sample)):
                        valider_list.append(dataset)
            valid_list = valider_list

        if valid_list:
            return random.sample(valid_list, 1)[0]
        
        return None

    def walkdir(self, rootdir):
        outlist = []
        for root, _, files in os.walk(rootdir):
            for name in files:
                outlist.append(root + "/" + name)
        return outlist
    
    def convert_to_EPSG4326(self, xy, dataset):
        """given a list of (x, y) encoded in dataset.crs, returns a list of (x, y) encoded in EPSG4326"""

        inproj, outproj = dataset.crs, Proj('EPSG:4326')

        return list(zip(*transform(inproj, outproj, *zip(*xy))))

    def checkdf(self, csv, outputfile):
        '''checks a csv with a "LATITUDE", and a "LONGITUDE" column, removes all rows that aren't in the dataset'''
        
        df = pd.read_csv(csv, index_col=False)
        columns = df.columns
        if "LATITUDE" not in columns or "LONGITUDE" not in columns:
            raise Exception("LATITUDE, and/or LONGITUDE not in csv")
        df_dict = df.to_dict(orient="list")

        keepindicies = []
        for i, (lat, lon) in enumerate(zip(df_dict["LATITUDE"], df_dict["LONGITUDE"])):
            if self.check(self.conv_coords(lat, lon)):
                keepindicies.append(i)
                print("keeping {}".format(i))
            
        outputdf = df.iloc[keepindicies]
        outputdf.to_csv(outputfile, index=False)

    def conv_coords(self, lat, lon):
        '''converts lat, lon to coords used by this class'''
        return {"LATITUDE": lat, "LONGITUDE": lon}

class FDSSCDataset(Dataset):
    '''Creates a dataset from a directory of datasets, and a csv of VALID locations, and crops out a location to feed into FDSSC

    The csv should have a "label" column, "LATITUDE", and "LONGITUDE"
    '''

    def __init__(self, datadir, csv, size=(9, 9), transform=None, lochecker=None):
        self.locations = pd.read_csv(csv)
        if lochecker:
            self.locationChecker = lochecker
        else:
            self.locationChecker = LocationChecker(datadir, rigorous=False)
        
        self.size = size
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.locations.loc[idx, "label"]
        coords = {"LATITUDE": self.locations.loc[idx, "LATITUDE"], "LONGITUDE": self.locations.loc[idx, "LONGITUDE"]}

        dataset_dir = self.locationChecker.check(coords)
        cubic = self.extract_cubic(dataset_dir, coords, size=self.size)

        if self.transform:
            cubic = self.transform(cubic)
        
        sample = {"label": label, "cubic": cubic}

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
